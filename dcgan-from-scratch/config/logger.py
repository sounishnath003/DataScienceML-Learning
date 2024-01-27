# ========================================
#  @project: dcgan-from-scratch
#  @filename: ~/Developer/dcgan-from-scratch/config/logger.py
#  @author: @github/sounishnath003
#  @generatedID: 4ed0c2a5-ae9d-4f65-9e73-c0b8084c6928
#  @createdAt: 27.01.2024 +05:30
# ========================================

import logging


class Logger:
    @staticmethod
    def get_logger():
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    "./logs/activity.log", mode="w+", encoding="utf-8", delay=True
                ),
            ],
            format="dcgan-torch:%(asctime)s:%(message)s",
        )

        return logging.getLogger()
