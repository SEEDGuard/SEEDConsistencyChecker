import os.path

from transformers import RobertaTokenizer
from transformers.models.t5 import T5ForConditionalGeneration

from core.CMIF.utils.instrumentation import load_jsonl, dump_jsonl


def predict_code_t5(inputPath, outputPath):
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    model1 = T5ForConditionalGeneration.from_pretrained("./core/CMIF/utils/t5_classification_final_ep2.mdl")
    data = load_jsonl(os.path.join(inputPath,"input.jsonl"))
    model1 = model1.to("cpu")
    index_c = 0

    predictions = []
    for test_element in data:
        test_source = test_element['source']
        inputs = tokenizer(test_source, return_tensors='pt').input_ids
        generated_ids = model1.generate(inputs.to("cpu"), num_beams=5, max_length=300, num_return_sequences=1)
        for i, beam_output in enumerate(generated_ids):
            fix = tokenizer.decode(beam_output, skip_special_tokens=True)
            predictions.append(
                {"source": test_source, "prediction": fix}
            )
        index_c += 1

    dump_jsonl(predictions, os.path.join(outputPath,"output.jsonl"))

