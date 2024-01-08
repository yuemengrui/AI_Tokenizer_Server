# *_*coding:utf-8 *_*
# @Author : YueMengRui
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class ErrorResponse(BaseModel):
    object: str = "error"
    errcode: int
    errmsg: str


class WordSegRequest(BaseModel):
    texts: List[str] = Field(description="需要分词的文本列表")
    return_origin: bool = Field(default=False, description="是否返回原始分词结果，不去除停用词, 默认false")
    return_completion: bool = Field(default=False, description="是否返回补全后的分词结果，默认false。")
    every_completion_limit: int = Field(default=5, description="每个关键词补全的数量")


class WordSegResponse(BaseModel):
    object: str = "word_seg"
    data: List


class KeywordsAddRequest(BaseModel):
    keywords_list: List = Field(description="关键词列表")
    clear: bool = Field(default=False, description="是否清空之前添加的关键词，再添加本次的关键词， 默认false")


class StopwordsAddRequest(BaseModel):
    stopwords_list: List = Field(description="停用词列表")
