import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidgetItem, QMessageBox, QStyle
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtGui
from v_todo_ui import Ui_MainWindow  # 引入生成的界面文件
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import opencc
from ckip_transformers.nlp import CkipWordSegmenter
import cn2an
import re
import sqlite3


def init_db():
    """初始化 SQLite 資料庫"""
    conn = sqlite3.connect("todo.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER,
            name TEXT,
            completed INTEGER DEFAULT 0,
            FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE
        )
    ''')
    conn.commit()
    conn.close()

# 初始化轉換器（s2t 代表簡體轉繁體）
converter = opencc.OpenCC('s2t')
# 初始化繁體中文分詞模型
ws_driver = CkipWordSegmenter(model="bert-base")

def convert_simplified_to_traditional(text):
    """將簡體中文轉為繁體"""
    return converter.convert(text)

def convert_chinese_numbers(text):
    """根據分詞結果，轉換時間格式及數字"""
    tokens = ws_driver([text])[0]
    
    converted_tokens = []
    chinese_number_pattern = re.compile(r'[零一二三四五六七八九十百千萬億]+')

    # 時間詞對應表
    time_mapping = {
        "一點": "1點", "二點": "2點", "兩點": "2點", "三點": "3點", "四點": "4點",
        "五點": "5點", "六點": "6點", "七點": "7點", "八點": "8點", "九點": "9點",
        "十點": "10點", "十一點": "11點", "十二點": "12點",
        "三十分": "30分", "十五分": "15分", "四十五分": "45分"
    }

    for i, raw_token in enumerate(tokens):
        # 先去除前後空白
        token = raw_token.strip()
        if not token:
            continue

        # 檢查是否在 time_mapping
        if token in time_mapping:
            converted_token = time_mapping[token]
        elif re.search(r"[點時分]", token):  # 檢查是否為時間詞
            parts = re.split(r"([點時分])", token)
            arabic_time_parts = []
            for part in parts:
                if chinese_number_pattern.fullmatch(part):
                    try:
                        arabic_time_parts.append(str(cn2an.transform(part, "cn2an")))
                    except ValueError:
                        arabic_time_parts.append(part)
                else:
                    arabic_time_parts.append(part)
            converted_token = "".join(arabic_time_parts)  # 重新組合
        elif chinese_number_pattern.fullmatch(token):
            try:
                converted_token = str(cn2an.transform(token, "cn2an"))
            except ValueError:
                converted_token = token
        else:
            converted_token = token

        converted_tokens.append(converted_token)

    converted_text = "".join(converted_tokens)

    # 最終格式化時間, ex: 4點30分 -> 4:30
    converted_text = re.sub(r'(\d+)點(\d+)分?', r'\1:\2', converted_text)
    converted_text = re.sub(r'(\d+)時(\d+)分?', r'\1:\2', converted_text)

    return converted_text

class ToDoApp(QMainWindow):
    def load_data(self):
        """從 SQLite 載入分類與項目"""
        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")

        # 載入分類
        cursor.execute("SELECT id, name FROM categories")
        categories = cursor.fetchall()
        self.ui.listWidgetCategories.clear()
        self.category_map = {}  # 建立 id -> 名稱 的映射
        for cat_id, name in categories:
            item = QListWidgetItem(name)
            self.ui.listWidgetCategories.addItem(item)
            self.category_map[name] = cat_id  # 儲存對應關係

        conn.close()

    def load_items_for_category(self, category_name):
        """根據分類名稱載入該分類下的所有項目，並更新 UI 與 self.item_map"""
        # 清空現有的項目 UI 與記憶體對應
        self.ui.listWidgetSubcategories.clear()
        self.item_map = {}  # 重新建立項目的映射

        # 根據分類名稱取得分類 ID
        category_id = self.category_map.get(category_name)
        if not category_id:
            return

        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("SELECT id, name, completed FROM items WHERE category_id = ?", (category_id,))
        items = cursor.fetchall()
        conn.close()

        # 將讀取到的項目加入 UI 與記憶體
        for item_id, name, completed in items:
            new_item = QListWidgetItem(name)
            new_item.setFlags(new_item.flags() | Qt.ItemIsUserCheckable)
            new_item.setCheckState(Qt.Checked if completed else Qt.Unchecked)
            self.ui.listWidgetSubcategories.addItem(new_item)
            self.item_map[name] = item_id

    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_connections()

        self.last_action = None  # 只記錄最近一次的可撤銷動作
        self.undo_timer = QTimer(self)  # 設置計時器
        self.undo_timer.setSingleShot(True)  # 只執行一次
        self.undo_timer.timeout.connect(self.clear_undo)  # 15秒後清除撤銷記錄

        # 初始狀態
        self.ui.stackedWidget.setCurrentWidget(self.ui.pageCategories)
        self.reset_editing_state()
        self.selected_category = None  # 目前選中的母分類項目
        self.selected_subcategory = None  # 目前選中的項目

        # 初始化記憶體中的映射：分類和項目
        self.category_map = {}
        self.item_map = {}

        # 從資料庫載入分類與項目
        self.load_data()

        # 初始化 Vosk 語音辨識
        self.model = Model("vosk-model-small-cn-0.22")  # 填入模型的路徑
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.audio_queue = queue.Queue()

        # 初始化按鈕圖示
        self.default_mic_icon = self.style().standardIcon(QStyle.SP_MediaVolumeMuted)  # 麥克風關閉
        self.recording_mic_icon = self.style().standardIcon(QStyle.SP_MediaVolume)  # 麥克風開啟

        self.ui.btnVoiceInputCategory.setIcon(self.default_mic_icon)
        self.ui.btnVoiceInputSubcategory.setIcon(self.default_mic_icon)
        self.ui.labelSpeechResult.setVisible(False) # 確保語音辨識結果區域一開始是隱藏的

        # 收音狀態
        self.is_recording = False

    def toggle_voice_input(self):
        """切換語音輸入（開始/停止）"""
        if not self.is_recording:
            self.start_voice_input()
        else:
            self.stop_voice_input()

    def start_voice_input(self):
        """開始語音輸入"""
        self.is_recording = True
        self.ui.statusbar.showMessage("正在收音...")
        self.ui.btnVoiceInputCategory.setIcon(self.recording_mic_icon)
        self.ui.btnVoiceInputSubcategory.setIcon(self.recording_mic_icon)

        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_queue.put(bytes(indata))

        self.stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                        channels=1, callback=callback)
        self.stream.start()

    def stop_voice_input(self):
        """停止語音輸入並立即處理辨識結果"""
        if not self.is_recording:
            return

        self.is_recording = False
        self.ui.statusbar.showMessage("收音已停止")
        self.ui.btnVoiceInputCategory.setIcon(self.default_mic_icon)
        self.ui.btnVoiceInputSubcategory.setIcon(self.default_mic_icon)

        if self.stream:
            self.stream.stop()
            self.stream.close()

        # 立即處理所有音訊數據，確保語音結果及時顯示
        self.process_audio_queue(force_finalize=True)

    def process_audio_queue(self, force_finalize=False):
        """處理所有音訊數據並進行語音辨識"""
        final_result = ""  # 存儲完整的語音結果

        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            if self.recognizer.AcceptWaveform(data):
                result_json = self.recognizer.Result()  # Vosk 回傳的是 JSON 字串
                result_dict = json.loads(result_json)   # 解析 JSON
                final_result = result_dict.get("text", "").strip()  # 提取辨識結果

        # 如果還沒有輸出結果，強制取得最終結果
        if force_finalize and not final_result:
            result_json = self.recognizer.FinalResult()  # 取得完整的最終辨識結果
            result_dict = json.loads(result_json)
            final_result = result_dict.get("text", "").strip()

        if final_result:
            # 簡轉繁 + 數字轉換
            traditional_text = convert_simplified_to_traditional(final_result)
            numeric_text = convert_chinese_numbers(traditional_text)

            # 更新 UI
            self.ui.labelSpeechResult.setText(f"語音辨識結果：{numeric_text}")
            self.ui.labelSpeechResult.setVisible(True)
            QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

            # 傳送數字轉換後的結果給 `process_voice_command()`
            self.process_voice_command({"text": numeric_text}) 
        else:
            self.ui.labelSpeechResult.setText("未識別到有效語音")
            self.ui.labelSpeechResult.setVisible(True)
            QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def process_voice_command(self, text):
        """處理語音輸入結果，使用 CKIP 進行分詞，並轉換適當的數字"""
        recognized_text = text.get("text", "").strip()

        if recognized_text:
            traditional_text = convert_simplified_to_traditional(recognized_text)
            numeric_text = convert_chinese_numbers(traditional_text)

            self.ui.labelSpeechResult.setText(f"語音辨識結果：{numeric_text}")
            self.ui.labelSpeechResult.setVisible(True)
            QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

            tokens = ws_driver([numeric_text])[0]
            print("語音分詞結果：", tokens)

            result = self.parse_command(tokens)

            # 若指令屬於項目操作，但目前不在項目頁面，則拒絕執行
            if result[0] in ["add_item", "delete_item", "edit_item", "complete_item"]:
                if self.ui.stackedWidget.currentWidget() != self.ui.pageSubcategories:
                    self.ui.labelSpeechResult.setText("請先進入項目頁面操作項目")
                    self.ui.labelSpeechResult.setVisible(True)
                    QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))
                    return

            if result[0] == "add_category" and result[1]:
                self.add_category_from_voice(result[1])
            elif result[0] == "delete_category" and result[1]:
                self.delete_category_from_voice(result[1])
            elif result[0] == "edit_category" and result[1] and result[2]:
                self.edit_category_from_voice(result[1], result[2])
            elif result[0] == "enter_category" and result[1]:
                self.enter_category_from_voice(result[1])
            elif result[0] == "add_item" and result[1]:
                self.add_item_from_voice(result[1])
            elif result[0] == "delete_item" and result[1]:
                self.delete_item_from_voice(result[1])
            elif result[0] == "edit_item" and result[1] and result[2]:
                self.edit_item_from_voice(result[1], result[2])
            elif result[0] == "complete_item" and result[1]:
                self.complete_item_from_voice(result[1])
            elif result[0] == "return_to_categories":
                self.return_to_categories()
            elif result[0] == "undo_last_action":
                self.undo_last_action()

            else:
                self.ui.labelSpeechResult.setText(f"無法識別的指令：{numeric_text}")
                self.ui.labelSpeechResult.setVisible(True)
                QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def parse_command(self, tokens):
        """解析語音分詞結果，轉換為指令與目標"""
        command = None
        target_old = ""
        target_new = ""

        # 統一異體字
        tokens = [token.replace("爲", "為") for token in tokens]

        # 去除每個 token 內部的空格
        tokens = [token.replace(" ", "") for token in tokens]

        # 關鍵詞片段定義（模糊匹配）
        return_fragments = {"返回", "回到", "上1頁", "前頁", "首頁"}
        complete_fragments = {"完成", "標記", "勾選", "打勾"}
        undo_fragment = {"撤銷", "復原"}

        # 判斷指令
        if "新增" in tokens and "分類" in tokens:
            command = "add_category"
        elif "刪除" in tokens and "分類" in tokens:
            command = "delete_category"
        elif "修改" in tokens and "分類" in tokens and "為" in tokens:
            command = "edit_category"
        elif "進入" in tokens and "分類" in tokens:
            command = "enter_category"
        elif "新增" in tokens and "項目" in tokens:
            command = "add_item"
        elif "刪除" in tokens and "項目" in tokens:
            command = "delete_item"
        elif "修改" in tokens and "項目" in tokens and "為" in tokens:
            command = "edit_item"
        elif any(fragment in token for token in tokens for fragment in complete_fragments):
            command = "complete_item"
        elif any(fragment in "".join(tokens) for fragment in return_fragments):
            command = "return_to_categories"
        elif any(fragment in "".join(tokens) for fragment in undo_fragment):
            command = "undo_last_action"

        # 找出名稱
        if command in ["edit_category", "edit_item"]:
            try:
                old_index = tokens.index("項目") + 1 if "項目" in tokens else tokens.index("分類") + 1
                new_index = tokens.index("為") + 1

                target_old = "".join(tokens[old_index:new_index-1])
                target_new = "".join(tokens[new_index:])
            except ValueError:
                pass  # 如果格式錯誤，則不做處理
        else:
            start_index = -1
            for i, token in enumerate(tokens):
                if token in ["新增", "刪除", "修改", "分類", "項目", "為", "進入", "返回", "完成", "標記", "勾選", "打勾"]:
                    continue
                if start_index == -1:
                    start_index = i
                target_old += token

        if command in ["edit_category", "edit_item"]:
            return command, target_old.strip(), target_new.strip()
        else:
            return command, target_old.strip()
    
    def add_category_from_voice(self, category_name):
        """透過語音新增分類，並存入 SQLite，同時支援撤銷"""
        # 檢查記憶體中的分類名稱，避免不必要的 SQL 操作
        if category_name in self.category_map:
            self.ui.labelSpeechResult.setText(f"分類「{category_name}」已存在")
            self.ui.labelSpeechResult.setVisible(True)
            QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))
            return

        # 直接插入 SQLite
        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("INSERT INTO categories (name) VALUES (?)", (category_name,))
        conn.commit()
        category_id = cursor.lastrowid  # 取得新分類的 ID
        conn.close()

        # 更新記憶體
        self.category_map[category_name] = category_id

        # 新增分類
        new_item = QListWidgetItem(category_name)
        self.ui.listWidgetCategories.addItem(new_item)

        # 覆蓋先前的撤銷動作
        self.last_action = ("add_category", category_name, category_id)
        self.reset_undo_timer()  # 重新計時

        self.ui.labelSpeechResult.setText(f"已新增分類：{category_name}")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def delete_category_from_voice(self, category_name):
        """透過語音刪除分類，並同步 SQLite"""
        if category_name not in self.category_map:
            self.ui.labelSpeechResult.setText(f"找不到分類：{category_name}")
            self.ui.labelSpeechResult.setVisible(True)
            QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))
            return

        category_id = self.category_map[category_name]

        # 刪除 UI 上的分類
        for index in range(self.ui.listWidgetCategories.count()):
            item = self.ui.listWidgetCategories.item(index)
            if item.text() == category_name:
                self.ui.listWidgetCategories.takeItem(index)
                break

        # 刪除 SQLite 紀錄
        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("DELETE FROM categories WHERE id = ?", (category_id,))
        conn.commit()
        conn.close()

        # 更新記憶體
        del self.category_map[category_name]

        # 支援撤銷
        self.last_action = ("delete_category", category_name, category_id)
        self.reset_undo_timer()

        self.ui.labelSpeechResult.setText(f"已刪除分類：{category_name}")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def edit_category_from_voice(self, old_category_name, new_category_name):
        """透過語音修改分類名稱，並確保名稱不重複"""
        if old_category_name not in self.category_map:
            self.ui.labelSpeechResult.setText(f"找不到分類：{old_category_name}")
            self.ui.labelSpeechResult.setVisible(True)
            QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))
            return

        if new_category_name in self.category_map:
            self.ui.labelSpeechResult.setText(f"分類「{new_category_name}」已存在，無法修改")
            self.ui.labelSpeechResult.setVisible(True)
            QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))
            return

        category_id = self.category_map[old_category_name]

        # 更新 SQLite
        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("UPDATE categories SET name = ? WHERE id = ?", (new_category_name, category_id))
        conn.commit()
        conn.close()

        # 更新 UI
        for index in range(self.ui.listWidgetCategories.count()):
            item = self.ui.listWidgetCategories.item(index)
            if item.text() == old_category_name:
                item.setText(new_category_name)
                break

        # 更新記憶體
        del self.category_map[old_category_name]
        self.category_map[new_category_name] = category_id

        # 支援撤銷
        self.last_action = ("edit_category", new_category_name, old_category_name, category_id)
        self.reset_undo_timer()

        self.ui.labelSpeechResult.setText(f"已將分類「{old_category_name}」修改為「{new_category_name}」")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def enter_category_from_voice(self, category_name):
        """透過語音進入分類"""
        for index in range(self.ui.listWidgetCategories.count()):
            item = self.ui.listWidgetCategories.item(index)
            if item.text() == category_name:
                self.selected_category = item
                self.ui.stackedWidget.setCurrentWidget(self.ui.pageSubcategories)
                self.ui.labelSpeechResult.setText(f"已進入分類：{category_name}")
                self.ui.labelSpeechResult.setVisible(True)
                QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))
                self.load_items_for_category(category_name)
                return

        self.ui.labelSpeechResult.setText(f"找不到分類：{category_name}")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def add_item_from_voice(self, item_name):
        """透過語音新增項目，並存入 SQLite"""
        if not self.selected_category:
            self.ui.labelSpeechResult.setText("請先選擇分類")
            self.ui.labelSpeechResult.setVisible(True)
            return

        category_name = self.selected_category.text()
        category_id = self.category_map.get(category_name)

        # 避免重複新增
        for index in range(self.ui.listWidgetSubcategories.count()):
            item = self.ui.listWidgetSubcategories.item(index)
            if item.text() == item_name:
                self.ui.labelSpeechResult.setText(f"項目「{item_name}」已存在")
                self.ui.labelSpeechResult.setVisible(True)
                return

        # 存入 SQLite
        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("INSERT INTO items (category_id, name, completed) VALUES (?, ?, ?)", (category_id, item_name, 0))
        conn.commit()
        item_id = cursor.lastrowid  # 取得新項目的 ID
        conn.close()

        # 更新 UI
        new_item = QListWidgetItem(item_name)
        new_item.setFlags(new_item.flags() | Qt.ItemIsUserCheckable)
        new_item.setCheckState(Qt.Unchecked)
        self.ui.listWidgetSubcategories.addItem(new_item)

        # 支援撤銷
        self.last_action = ("add_item", item_name, category_name, item_id)
        self.reset_undo_timer()

        self.ui.labelSpeechResult.setText(f"已新增項目：{item_name}")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def delete_item_from_voice(self, item_name):
        """透過語音刪除項目，並同步 SQLite"""
        if not self.selected_category:
            self.ui.labelSpeechResult.setText("請先選擇分類")
            self.ui.labelSpeechResult.setVisible(True)
            return

        category_name = self.selected_category.text()
        category_id = self.category_map.get(category_name)
        item_id = None

        # 刪除 UI 中的項目
        for index in range(self.ui.listWidgetSubcategories.count()):
            item = self.ui.listWidgetSubcategories.item(index)
            if item.text() == item_name:
                self.ui.listWidgetSubcategories.takeItem(index)
                break
        else:
            self.ui.labelSpeechResult.setText(f"找不到項目：{item_name}")
            self.ui.labelSpeechResult.setVisible(True)
            return

        # 從 SQLite 刪除
        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("SELECT id FROM items WHERE category_id = ? AND name = ?", (category_id, item_name))
        row = cursor.fetchone()
        if row:
            item_id = row[0]
            cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
            conn.commit()
        conn.close()

        # 支援撤銷
        if item_id:
            self.last_action = ("delete_item", item_name, category_name, item_id)
            self.reset_undo_timer()

        self.ui.labelSpeechResult.setText(f"已刪除項目：{item_name}")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def edit_item_from_voice(self, old_name, new_name):
        """透過語音修改項目名稱，並確保名稱不重複"""
        if not self.selected_category:
            self.ui.labelSpeechResult.setText("請先選擇分類")
            self.ui.labelSpeechResult.setVisible(True)
            return

        category_name = self.selected_category.text()
        category_id = self.category_map.get(category_name)

        # 檢查新名稱是否重複
        for index in range(self.ui.listWidgetSubcategories.count()):
            item = self.ui.listWidgetSubcategories.item(index)
            if item.text() == new_name:
                self.ui.labelSpeechResult.setText(f"項目「{new_name}」已存在")
                self.ui.labelSpeechResult.setVisible(True)
                return

        item_id = self.item_map.get(old_name)

        # 更新 SQLite
        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("UPDATE items SET name = ? WHERE category_id = ? AND name = ?", (new_name, category_id, old_name))
        conn.commit()
        conn.close()

        # 更新 UI
        for index in range(self.ui.listWidgetSubcategories.count()):
            item = self.ui.listWidgetSubcategories.item(index)
            if item.text() == old_name:
                item.setText(new_name)
                break

        # 支援撤銷
        self.last_action = ("edit_item", new_name, old_name, category_name, item_id)

        self.reset_undo_timer()

        self.ui.labelSpeechResult.setText(f"已修改項目：{old_name} → {new_name}")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def complete_item_from_voice(self, item_name):
        """透過語音標記項目為完成，並存入 SQLite"""
        for index in range(self.ui.listWidgetSubcategories.count()):
            item = self.ui.listWidgetSubcategories.item(index)
            if item.text() == item_name:
                item.setCheckState(Qt.Checked)  # 標記為完成
                self.ui.labelSpeechResult.setText(f"已標記完成：{item_name}")
                self.ui.labelSpeechResult.setVisible(True)

                # 更新 SQLite
                conn = sqlite3.connect("todo.db")
                cursor = conn.cursor()
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.execute("UPDATE items SET completed = 1 WHERE name = ?", (item_name,))
                conn.commit()
                conn.close()

                # 支援撤銷
                self.last_action = ("uncomplete_item", item_name)
                self.reset_undo_timer()

                QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))
                return

        self.ui.labelSpeechResult.setText(f"找不到項目：{item_name}")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def return_to_categories(self):
        """返回分類主頁"""
        self.ui.stackedWidget.setCurrentWidget(self.ui.pageCategories)  # 回到分類主頁
        self.ui.labelSpeechResult.setText("已返回分類頁面")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def undo_last_action(self):
        """回到上一個動作，並同步 SQLite"""
        if not self.last_action:
            self.ui.labelSpeechResult.setText("沒有可撤銷的動作")
            self.ui.labelSpeechResult.setVisible(True)
            QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))
            return

        action = self.last_action
        self.last_action = None
        self.undo_timer.stop()  # 停止計時

        conn = sqlite3.connect("todo.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")

        if action[0] == "delete_category":
            category_name, category_id = action[1], action[2]
            cursor.execute("INSERT INTO categories (id, name) VALUES (?, ?)", (category_id, category_name))
            conn.commit()

            new_item = QListWidgetItem(category_name)
            self.ui.listWidgetCategories.addItem(new_item)
            self.category_map[category_name] = category_id

        elif action[0] == "add_category":
            category_name, category_id = action[1], action[2]
            cursor.execute("DELETE FROM categories WHERE id = ?", (category_id,))
            conn.commit()

            for index in range(self.ui.listWidgetCategories.count()):
                item = self.ui.listWidgetCategories.item(index)
                if item.text() == category_name:
                    self.ui.listWidgetCategories.takeItem(index)
                    break

            del self.category_map[category_name]

        elif action[0] == "edit_category":
            new_name, old_name, category_id = action[1], action[2], action[3]
            cursor.execute("UPDATE categories SET name = ? WHERE id = ?", (old_name, category_id))
            conn.commit()

            for index in range(self.ui.listWidgetCategories.count()):
                item = self.ui.listWidgetCategories.item(index)
                if item.text() == new_name:
                    item.setText(old_name)
                    break

            del self.category_map[new_name]
            self.category_map[old_name] = category_id

        elif action[0] == "add_item":
            item_name, category_name, item_id = action[1], action[2], action[3]
            cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
            conn.commit()

            # 更新 UI
            for index in range(self.ui.listWidgetSubcategories.count()):
                item = self.ui.listWidgetSubcategories.item(index)
                if item.text() == item_name:
                    self.ui.listWidgetSubcategories.takeItem(index)
                    break

            # 更新記憶體
            if item_name in self.item_map:
                del self.item_map[item_name]

        elif action[0] == "delete_item":
            item_name, category_name, item_id = action[1], action[2], action[3]
            category_id = self.category_map.get(category_name)
            # 檢查該 id 是否已存在
            cursor.execute("SELECT id FROM items WHERE id = ?", (item_id,))
            if cursor.fetchone() is None:
                cursor.execute("INSERT INTO items (id, category_id, name, completed) VALUES (?, ?, ?, ?)", 
                               (item_id, category_id, item_name, 0))
                conn.commit()
            # 更新 UI：重新將該項目加回
            new_item = QListWidgetItem(item_name)
            new_item.setFlags(new_item.flags() | Qt.ItemIsUserCheckable)
            new_item.setCheckState(Qt.Unchecked)
            self.ui.listWidgetSubcategories.addItem(new_item)
            # 更新記憶體
            self.item_map[item_name] = item_id

        elif action[0] == "edit_item":
            # 
            new_name, old_name, category_name, item_id = action[1], action[2], action[3], action[4]
            category_id = self.category_map.get(category_name)
            cursor.execute("UPDATE items SET name = ? WHERE category_id = ? AND id = ?", 
                   (old_name, category_id, item_id))
            conn.commit()

            # 更新 UI
            for index in range(self.ui.listWidgetSubcategories.count()):
                item = self.ui.listWidgetSubcategories.item(index)
                if item.text() == new_name:
                    item.setText(old_name)
                    break
                
            # 更新記憶體
            if new_name in self.item_map:
                del self.item_map[new_name]
            self.item_map[old_name] = item_id

        conn.close()
        self.ui.labelSpeechResult.setText("已撤銷上一個動作")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def reset_undo_timer(self):
        """重新開始15秒撤銷計時"""
        self.undo_timer.start(15000)  # 15秒後清除撤銷記錄

    def clear_undo(self):
        """清除撤銷動作（15秒後執行）"""
        self.last_action = None
        self.ui.labelSpeechResult.setText("撤銷時間已過，無法回復")
        self.ui.labelSpeechResult.setVisible(True)
        QTimer.singleShot(2000, lambda: self.ui.labelSpeechResult.setVisible(False))

    def setup_connections(self):
        # 第一層按鈕
        self.ui.btnAddCategory.clicked.connect(self.add_category)
        self.ui.btnEditCategory.clicked.connect(self.edit_category)
        self.ui.btnDeleteCategory.clicked.connect(self.delete_category)
        self.ui.btnManageItems.clicked.connect(self.manage_items)

        # 語音輸入按鈕（第一層）
        self.ui.btnVoiceInputCategory.clicked.connect(self.toggle_voice_input)

        # 第二層按鈕
        self.ui.btnAddSubcategory.clicked.connect(self.add_subcategory)
        self.ui.btnEditSubcategory.clicked.connect(self.edit_subcategory)
        self.ui.btnDeleteSubcategory.clicked.connect(self.delete_subcategory)
        self.ui.btnBackToCategories.clicked.connect(self.back_to_categories)

        # 語音輸入按鈕（第二層）
        self.ui.btnVoiceInputSubcategory.clicked.connect(self.toggle_voice_input)

        # 監聽項目打勾狀態變化
        self.ui.listWidgetSubcategories.itemChanged.connect(self.toggle_completed_status)

        # 確認/取消按鈕
        self.ui.btnConfirmEdit.clicked.connect(self.confirm_edit_task)
        self.ui.btnCancelEdit.clicked.connect(self.cancel_edit_task)
        self.ui.btnConfirmEdit_2.clicked.connect(self.confirm_edit_task_2)
        self.ui.btnCancelEdit_2.clicked.connect(self.cancel_edit_task)

        # 列表點擊事件
        self.ui.listWidgetCategories.itemClicked.connect(self.select_category)
        self.ui.listWidgetSubcategories.itemClicked.connect(self.select_subcategory)

    # 第一層功能
    def add_category(self):
        self.ui.textEditEditTask.setVisible(True)
        self.ui.btnConfirmEdit.setVisible(True)
        self.ui.btnCancelEdit.setVisible(True)
        self.ui.textEditEditTask.clear()
        self.edit_mode = "add_category"

    def edit_category(self):
        if not self.selected_category:
            QMessageBox.warning(self, "提示", "請選擇一個分類")
            return
        self.ui.textEditEditTask.setVisible(True)
        self.ui.btnConfirmEdit.setVisible(True)
        self.ui.btnCancelEdit.setVisible(True)
        self.ui.textEditEditTask.setText(self.selected_category.text())
        self.edit_mode = "edit_category"

    def delete_category(self):
        if not self.selected_category:
            QMessageBox.warning(self, "提示", "請選擇一個分類")
            return
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("刪除確認")
        msg_box.setText("確定刪除此分類？")
        # 自訂按鈕
        btn_yes = msg_box.addButton("是", QMessageBox.YesRole)
        btn_no = msg_box.addButton("否", QMessageBox.NoRole)

        msg_box.setDefaultButton(btn_no)
        msg_box.exec()

        if msg_box.clickedButton() == btn_yes:
            category_name = self.selected_category.text()
            category_id = self.category_map.get(category_name)
            # 從 SQLite 刪除
            conn = sqlite3.connect("todo.db")
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("DELETE FROM categories WHERE id = ?", (category_id,))
            conn.commit()
            conn.close()

            # 從 UI 刪除
            self.ui.listWidgetCategories.takeItem(self.ui.listWidgetCategories.row(self.selected_category))
            del self.category_map[category_name]

            # 記錄撤銷操作
            self.last_action = ("add_category", category_name, category_id)
            self.reset_undo_timer()

            self.selected_category = None
            self.reset_editing_state()

    def manage_items(self):
        if not self.selected_category:
            QMessageBox.warning(self, "提示", "請選擇一個分類")
            return
        self.load_items_for_category(self.selected_category.text())
        self.ui.stackedWidget.setCurrentWidget(self.ui.pageSubcategories)

    # 第二層功能
    def add_subcategory(self):
        self.ui.textEditEditTask_2.setVisible(True)
        self.ui.btnConfirmEdit_2.setVisible(True)
        self.ui.btnCancelEdit_2.setVisible(True)
        self.ui.textEditEditTask_2.clear()
        self.edit_mode = "add_subcategory"

    def edit_subcategory(self):
        if not self.selected_subcategory:
            QMessageBox.warning(self, "提示", "請選擇一個項目")
            return
        self.ui.textEditEditTask_2.setVisible(True)
        self.ui.btnConfirmEdit_2.setVisible(True)
        self.ui.btnCancelEdit_2.setVisible(True)
        self.ui.textEditEditTask_2.setText(self.selected_subcategory.text())
        self.edit_mode = "edit_subcategory"

    def delete_subcategory(self):
        if not self.selected_subcategory:
            QMessageBox.warning(self, "提示", "請選擇一個項目")
            return
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("刪除確認")
        msg_box.setText("確定刪除此項目？")
        btn_yes = msg_box.addButton("是", QMessageBox.YesRole)
        btn_no = msg_box.addButton("否", QMessageBox.NoRole)
        msg_box.setDefaultButton(btn_no)
        msg_box.exec()
        
        if msg_box.clickedButton() == btn_yes:
            item_name = self.selected_subcategory.text()
            category_name = self.selected_category.text()
            category_id = self.category_map.get(category_name)

            # 從 SQLite 刪除，先取得 item_id
            conn = sqlite3.connect("todo.db")
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("SELECT id FROM items WHERE category_id = ? AND name = ?", (category_id, item_name))
            row = cursor.fetchone()
            if row:
                item_id = row[0]
                cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
                conn.commit()
                # 更新記憶體
                if item_name in self.item_map:
                    del self.item_map[item_name]
            conn.close()

            # 從 UI 刪除
            self.ui.listWidgetSubcategories.takeItem(self.ui.listWidgetSubcategories.row(self.selected_subcategory))
            self.selected_subcategory = None
            self.reset_editing_state()

    def toggle_completed_status(self, item):
        """當使用者勾選 CheckBox 時，改變項目的樣式，同時更新 SQLite 資料庫"""
        if item.checkState() == Qt.Checked:
            # 已完成：改變字體顏色為灰色，並加刪除線
            font = item.font()
            font.setStrikeOut(True)
            item.setFont(font)
            item.setForeground(QtGui.QColor("gray"))
            completed = 1
        else:
            # 未完成：恢復正常字體
            font = item.font()
            font.setStrikeOut(False)
            item.setFont(font)
            item.setForeground(QtGui.QColor("black"))
            completed = 0
        # 使用 item_map 根據 item 名稱取得對應的資料庫 id
        item_id = self.item_map.get(item.text())
        if item_id is not None:
            conn = sqlite3.connect("todo.db")
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("UPDATE items SET completed = ? WHERE id = ?", (completed, item_id))
            conn.commit()
            conn.close()

    def back_to_categories(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.pageCategories)

    # 確認/取消按鈕
    def confirm_edit_task(self):
        new_text = self.ui.textEditEditTask.toPlainText().strip()
        if not new_text:
            QMessageBox.warning(self, "提示", "輸入不能為空")
            return

        if self.edit_mode == "add_category":
            # 新增分類到 SQLite
            conn = sqlite3.connect("todo.db")
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            try:
                cursor.execute("INSERT INTO categories (name) VALUES (?)", (new_text,))
                conn.commit()
                category_id = cursor.lastrowid  # 取得新分類的ID
            except sqlite3.IntegrityError:
                QMessageBox.warning(self, "提示", f"分類「{new_text}」已存在")
                conn.close()
                return
            conn.close()

            # 更新記憶體與 UI
            self.category_map[new_text] = category_id
            self.ui.listWidgetCategories.addItem(QListWidgetItem(new_text))

            # 記錄撤銷操作
            self.last_action = ("delete_category", new_text, category_id)
            self.reset_undo_timer()

        elif self.edit_mode == "edit_category" and self.selected_category:
            # 修改分類：更新 SQLite
            old_text = self.selected_category.text()
            category_id = self.category_map.get(old_text)
            conn = sqlite3.connect("todo.db")
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("UPDATE categories SET name = ? WHERE id = ?", (new_text, category_id))
            conn.commit()
            conn.close()

            # 更新 UI 與記憶體
            self.selected_category.setText(new_text)
            del self.category_map[old_text]
            self.category_map[new_text] = category_id

            self.last_action = ("edit_category", new_text, old_text, category_id)
            self.reset_undo_timer()

        self.reset_editing_state()

    def confirm_edit_task_2(self):
        new_text = self.ui.textEditEditTask_2.toPlainText().strip()
        if not new_text:
            QMessageBox.warning(self, "提示", "輸入不能為空")
            return

        if self.edit_mode == "add_subcategory":
            # 取得目前分類的ID
            if not self.selected_category:
                QMessageBox.warning(self, "提示", "請先選擇一個分類")
                return
            category_name = self.selected_category.text()
            category_id = self.category_map.get(category_name)

            # 存入 SQLite
            conn = sqlite3.connect("todo.db")
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("INSERT INTO items (category_id, name, completed) VALUES (?, ?, ?)", (category_id, new_text, 0))
            conn.commit()
            item_id = cursor.lastrowid
            conn.close()

            # 更新 UI 與記憶體
            item = QListWidgetItem(new_text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.ui.listWidgetSubcategories.addItem(item)
            self.item_map[new_text] = item_id

            # 記錄撤銷操作
            self.last_action = ("delete_item", new_text, category_name, item_id)
            self.reset_undo_timer()

        elif self.edit_mode == "edit_subcategory" and self.selected_subcategory:
            # 修改項目：更新 SQLite
            old_text = self.selected_subcategory.text()
            category_name = self.selected_category.text()
            category_id = self.category_map.get(category_name)

            conn = sqlite3.connect("todo.db")
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("UPDATE items SET name = ? WHERE category_id = ? AND name = ?", (new_text, category_id, old_text))
            conn.commit()
            conn.close()

            # 更新 UI 與記憶體
            self.selected_subcategory.setText(new_text)
            del self.item_map[old_text]
            self.item_map[new_text] = None  # 若能取得ID，可更新；否則後續需要重新載入

            self.last_action = ("edit_item", new_text, old_text, category_name)
            self.reset_undo_timer()

        self.reset_editing_state()

    def cancel_edit_task(self):
        self.reset_editing_state()

    # 選中事件
    def select_category(self, item):
        self.selected_category = item
        self.ui.btnEditCategory.setEnabled(True)
        self.ui.btnDeleteCategory.setEnabled(True)
        self.ui.btnManageItems.setEnabled(True)

    def select_subcategory(self, item):
        self.selected_subcategory = item
        self.ui.btnEditSubcategory.setEnabled(True)
        self.ui.btnDeleteSubcategory.setEnabled(True)

    # 重置狀態
    def reset_editing_state(self):
        self.ui.textEditEditTask.setVisible(False)
        self.ui.btnConfirmEdit.setVisible(False)
        self.ui.btnCancelEdit.setVisible(False)
        self.ui.btnEditCategory.setEnabled(False)
        self.ui.btnDeleteCategory.setEnabled(False)
        self.ui.btnManageItems.setEnabled(False)
        self.ui.textEditEditTask_2.setVisible(False)
        self.ui.btnConfirmEdit_2.setVisible(False)
        self.ui.btnCancelEdit_2.setVisible(False)
        self.ui.btnEditSubcategory.setEnabled(False)
        self.ui.btnDeleteSubcategory.setEnabled(False)
        self.edit_mode = None

if __name__ == "__main__":
    init_db()
    app = QApplication(sys.argv)
    window = ToDoApp()
    window.show()
    sys.exit(app.exec_())
