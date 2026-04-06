from paddleocr import TableRecognitionPipelineV2

pipeline = TableRecognitionPipelineV2()
# ocr = TableRecognitionPipelineV2(use_doc_orientation_classify=True) # Specify whether to use the document orientation classification model with use_doc_orientation_classify
# ocr = TableRecognitionPipelineV2(use_doc_unwarping=True) # Specify whether to use the text image unwarping module with use_doc_unwarping
# ocr = TableRecognitionPipelineV2(device="gpu") # Specify the device to use GPU for model inference
output = pipeline.predict("test_files/test_gizi_2.png") # Predict the structured output of the input image
for res in output:
    res.print() ## Print the predicted structured output
    res.save_to_img("./output/")
    res.save_to_xlsx("./output/")
    res.save_to_html("./output/")
    res.save_to_json("./output/")