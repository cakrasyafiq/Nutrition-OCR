from pathlib import Path
from paddleocr import TableRecognitionPipelineV2

BASE_DIR = Path(__file__).resolve().parents[1]
input_image = BASE_DIR / "test_files" / "test_gizi_2.png"
output_dir = BASE_DIR / "output"
output_dir.mkdir(parents=True, exist_ok=True)

pipeline = TableRecognitionPipelineV2()
# ocr = TableRecognitionPipelineV2(use_doc_orientation_classify=True) # Specify whether to use the document orientation classification model with use_doc_orientation_classify
# ocr = TableRecognitionPipelineV2(use_doc_unwarping=True) # Specify whether to use the text image unwarping module with use_doc_unwarping
# ocr = TableRecognitionPipelineV2(device="gpu") # Specify the device to use GPU for model inference
output = pipeline.predict(str(input_image))
for res in output:
    res.print()
    res.save_to_img(str(output_dir))
    res.save_to_xlsx(str(output_dir))
    res.save_to_html(str(output_dir))
    res.save_to_json(str(output_dir))