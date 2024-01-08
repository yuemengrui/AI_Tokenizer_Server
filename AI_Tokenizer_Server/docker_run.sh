#!/bin/bash

cd /workspace/AI_Tokenizer_Server && CUDA_VISIBLE_DEVICES=0 nohup python manage_ai_tokenizer_servers.py >/dev/null 2>&1 &
echo "server runing"
/bin/bash
