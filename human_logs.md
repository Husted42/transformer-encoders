I created this manual log file to descrivbe the sub-results we get, hopefully this will make it easier to understand design choises and remember what we did for the report

# DPR
### Log 1
Trained the DPR

epoch : 30
batch_size : 32
learning_rate : 0.001

'''
python main.py retriever=dpr model.query_encoder_weights_path=outputs/dpr/checkpoints/query_encoder.pt model.document_encoder_weights_path=outputs/dpr/checkpoints/document_encoder.pt
'''

I got very bad results:
Epoch 30/30: training loss 2.7733

Train results: {'recall': 0.000346969696969697}
Val results: {'recall': 0.0}

In classic ML fashion let's see if we can overfit to a small amount of data: <br>
'''
python -m unittest tests\debugging\test_fiqa_overfit.py -v
'''

This returned a loss of 0.0003, so we are able to overfit.

It could be the pooling method? Maybe we should try mean instead of cls:

'''
python tests\debugging\test_fiqa_overfit.py
'''

### Log 2
We redo the same expirement again, buit print the grad, loss and positive/negative scores.
 - Positve shows scores for relevant passages
 - Negative show scores fir other passages

And want relevant passages to have a high score and negative to have a low score

'''
Epoch 2/8: training loss 2.8127, positive -9.6515, negative -9.6536, query grad 1.1190, document grad 1.5498
Epoch 3/8: training loss 2.8015, positive -4.1050, negative -4.1036, query grad 0.8749, document grad 1.1913
Epoch 4/8: training loss 2.7939, positive -2.9365, negative -2.9364, query grad 0.7398, document grad 0.9760
Epoch 5/8: training loss 2.7904, positive -1.9900, negative -1.9897, query grad 0.6573, document grad 0.7676
'''


Postive/negative scores are nearly identical meaning that the model isn't seperating them. 

### Log 3
'''
python train.py retriever=dpr model.pooling_type=mean training.learning_rate=0.0001 training.epochs=8 training.output_dir=outputs/dpr/mean-0001-run
'''

Results:
'''
Epoch 1: validation recall@10=0.0000 (128 queries)
Epoch 2/8: training loss 2.6215, positive 9.1097, negative 8.7628, query grad 2.9777, document grad 2.3636
Epoch 3/8: training loss 2.4529, positive 20.9317, negative 20.1605, query grad 4.3939, document grad 3.1298
Epoch 4/8: training loss 2.2315, positive 26.1456, negative 24.7741, query grad 5.6087, document grad 4.0730
Epoch 5/8: training loss 2.0098, positive 27.7810, negative 25.7729, query grad 6.6941, document grad 4.9013
Saved epoch 5 encoder checkpoints to outputs\dpr\mean-0001-run\epoch_005
Building the full document index for validation at epoch 5...
Epoch 5: validation recall@10=0.0184 (128 queries)
Epoch 6/8: training loss 1.7979, positive 30.4583, negative 27.7865, query grad 7.5092, document grad 5.8086
Epoch 7/8: training loss 1.6163, positive 29.5459, negative 26.3194, query grad 8.2525, document grad 6.4060
Epoch 8/8: training loss 1.4464, positive 30.9165, negative 27.0705, query grad 8.7440, document grad 6.9704
Saved epoch 8 encoder checkpoints to outputs\dpr\mean-0001-run\epoch_008
Building the full document index for validation at epoch 8...
Epoch 8: validation recall@10=0.0204 (128 queries)
'''

This seems to be an improvement, but validation retrieval is still low and the difference is small.

If we look at Epoch 5 and 8 we get a improvent of 2.4, but still very low.

'''
Epoch 5 validation: {'recall': 0.007222222222222222}
Epoch 8 validation: {'recall': 0.017483333333333333}
'''

### Log 4
'''
python train.py retriever=dpr model.pooling_type=mean training.learning_rate=0.0001 training.epochs=16 training.validation_max_queries=500 training.output_dir=outputs/dpr/mean-0001-16epochs
'''

| Epoch | Training loss | Positive | Negative | Query grad | Document grad | Validation recall@10 |
|------:|--------------:|---------:|---------:|-----------:|--------------:|---------------------:|
| 1 | — | — | — | — | — | 0.0002 |
| 2 | 2.6215 | 9.1097 | 8.7628 | 2.9777 | 2.3636 | — |
| 3 | 2.4529 | 20.9317 | 20.1605 | 4.3939 | 3.1298 | — |
| 4 | 2.2315 | 26.1456 | 24.7741 | 5.6087 | 4.0730 | — |
| 5 | 2.0098 | 27.7810 | 25.7729 | 6.6941 | 4.9013 | 0.0072 |
| 6 | 1.7979 | 30.4583 | 27.7865 | 7.5092 | 5.8086 | — |
| 7 | 1.6163 | 29.5459 | 26.3194 | 8.2525 | 6.4060 | — |
| 8 | 1.4464 | 30.9165 | 27.0705 | 8.7440 | 6.9704 | — |
| 9 | 1.2927 | 31.3862 | 26.9109 | 8.9866 | 7.2552 | — |
| 10 | 1.1528 | 31.4466 | 26.3033 | 9.2895 | 7.7667 | 0.0123 |
| 11 | 1.0465 | 31.5695 | 25.9922 | 9.3493 | 7.8882 | — |
| 12 | 0.9563 | 31.7304 | 25.7092 | 9.5026 | 8.1884 | — |
| 13 | 0.8717 | 31.6924 | 25.1554 | 9.5487 | 8.3002 | — |
| 14 | 0.7510 | 31.1321 | 23.9284 | 9.0778 | 8.0264 | — |
| 15 | 0.7384 | 31.9024 | 24.4866 | 9.4005 | 8.3487 | 0.0159 |

It seems like the model is learning loss is decreassing and the gap is increassing. Let's view the results

'''
python main.py retriever=dpr model.pooling_type=mean model.query_encoder_weights_path=outputs\dpr\mean-0001-16epochs\query_encoder.pt model.document_encoder_weights_path=outputs\dpr\mean-0001-16epochs\document_encoder.pt
'''

'''
Train results: {'recall': 0.18671349206349266}
Val results: {'recall': 0.02045555555555556}
'''

Maybe it's time to try with more Epochs. I will rerun with 32.