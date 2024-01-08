# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os

FASTAPI_TITLE = 'AI_Tokenizer_Servers'
FASTAPI_HOST = '0.0.0.0'
FASTAPI_PORT = 24612

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
TEMP = './temp'
os.makedirs(TEMP, exist_ok=True)

CHINESE_WORD_SEGMENTATION_MODEL_PATH = '/workspace/Models/ch_word_segmentation_coarse'
KEYEDVECTORS_MODEL_PATH = '/workspace/Models/sgns.merge.bigram.bz2'

# API LIMIT
API_LIMIT = {
    "base": "600/minute",
}
