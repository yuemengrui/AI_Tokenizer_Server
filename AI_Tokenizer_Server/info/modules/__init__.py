# *_*coding:utf-8 *_*
from fastapi import FastAPI
from . import word_seg, health


def register_router(app: FastAPI):
    app.include_router(router=word_seg.router, prefix="", tags=["Word Segmentation"])
    app.include_router(router=health.router, prefix="", tags=["health"])
