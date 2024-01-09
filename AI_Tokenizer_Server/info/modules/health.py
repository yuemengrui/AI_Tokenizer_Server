# *_*coding:utf-8 *_*
# @Author : YueMengRui
from fastapi import APIRouter, Request
from info import limiter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.api_route('/ai/health', methods=['GET'], summary="health")
@limiter.limit("6/minute")
def health(request: Request):
    return JSONResponse({'msg': 'I am very healthy'})
