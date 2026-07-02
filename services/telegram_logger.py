import os
import requests
import json
from datetime import datetime

class TelegramLogger:
    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)

    def _send_message(self, text: str):
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            print(f"⚠️ Failed to send Telegram log: {e}")

    def log_data_sync(self, users_count: int, matches_count: int, time_taken: float, total_records: int):
        text = (
            f"📈 <b>[DATA SYNC THÀNH CÔNG]</b>\n\n"
            f"👤 Số lượng User lấy về: <b>{users_count}</b>\n"
            f"🤝 Số lượng Match/Swipe: <b>{matches_count}</b>\n"
            f"💾 Tổng kích thước Dataset (ước tính): <b>{total_records} records</b>\n"
            f"⏳ Thời gian crawl: <b>{time_taken:.2f} giây</b>"
        )
        self._send_message(text)

    def log_training_start(self, model_name: str, config: dict = None):
        config_str = "\n".join([f"- {k}: <code>{v}</code>" for k, v in (config or {}).items()])
        text = (
            f"⚙️ <b>[ĐANG BẮT ĐẦU TRAIN MODEL...]</b>\n\n"
            f"🧠 Model: <b>{model_name}</b>\n"
            f"🕰 Bắt đầu lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Cấu hình:\n{config_str}"
        )
        self._send_message(text)

    def log_training_end(self, version: str, time_taken: float, total_pairs: int, train_size: int, test_size: int, auc: float, acc: float, status: str, filepath: str):
        text = (
            f"✅ <b>[TRAINING HOÀN TẤT]</b>\n\n"
            f"📦 <b>Version:</b> <code>{version}</code>\n"
            f"⏳ <b>Thời gian chạy:</b> {time_taken:.2f} giây\n"
            f"📈 <b>Data Size:</b> {total_pairs} pairs\n"
            f"   ├─ Train: {train_size}\n"
            f"   └─ Test: {test_size}\n"
            f"🎯 <b>Metrics:</b>\n"
            f"   ├─ ROC-AUC: <b>{auc:.4f}</b>\n"
            f"   └─ Accuracy: <b>{acc:.4f}</b>\n"
            f"🛡️ <b>Status:</b> <b>{status}</b>\n\n"
            f"🤖 <b>Checkpoint:</b> <code>{filepath}</code>"
        )
        self._send_message(text)

    def log_alert(self, message: str):
        text = f"🚨 <b>[CẢNH BÁO MLOPS]</b>\n\n{message}"
        self._send_message(text)

# Khởi tạo instance mặc định
telegram_logger = TelegramLogger()
