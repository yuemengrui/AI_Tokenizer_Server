# *_*coding:utf-8 *_*
# @Author : YueMengRui
from mylogger import logger
from fastapi import APIRouter, Request
from info import limiter, chinese_word_seg_model, keyed_vec_model
from configs import API_LIMIT
from copy import deepcopy
from .protocol import WordSegRequest, WordSegResponse, KeywordsAddRequest, StopwordsAddRequest
from fastapi.responses import JSONResponse

router = APIRouter()


@router.api_route('/ai/tokenize/word_seg', methods=['POST'], response_model=WordSegResponse, summary="Word Seg")
@limiter.limit(API_LIMIT['base'])
def chinese_word_seg(request: Request,
                     req: WordSegRequest,
                     ):
    logger.info(req.dict())

    res = chinese_word_seg_model.predict(req.texts, return_origin=req.return_origin)
    if req.return_completion:
        resp = []
        for one_sen in res:
            temp = deepcopy(one_sen)
            for one_word in one_sen:
                sim_list = keyed_vec_model.similar_by_key(one_word, topn=req.every_completion_limit)
                temp.extend([x[0] for x in sim_list])

            resp.append(list(set(deepcopy(temp))))

        res = resp

    return JSONResponse(WordSegResponse(data=res).dict())


@router.api_route('/ai/tokenize/keywords/add', methods=['POST'], summary="Word Seg keywords add")
@limiter.limit(API_LIMIT['base'])
def chinese_word_seg_keywords_add(request: Request,
                                  req: KeywordsAddRequest,
                                  ):
    logger.info(req.dict())

    chinese_word_seg_model.add_keywords_combine(req.keywords_list, clear=req.clear)

    return JSONResponse({'msg': u'成功'})


@router.api_route('/ai/tokenize/stopwords/add', methods=['POST'], summary="Word Seg stopwords add")
@limiter.limit(API_LIMIT['base'])
def chinese_word_seg_stopwords_add(request: Request,
                                   req: StopwordsAddRequest,
                                   ):
    logger.info(req.dict())

    chinese_word_seg_model.add_stopwords(req.stopwords_list)

    return JSONResponse({'msg': u'成功'})
