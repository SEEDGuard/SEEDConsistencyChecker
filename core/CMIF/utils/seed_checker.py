import logging
import os.path

from transformers import RobertaTokenizer
from transformers.models.t5 import T5ForConditionalGeneration

from core.CMIF.utils.seed_processor import load_jsonl, dump_jsonl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def predict(DATA_DIR, DEST_DIR):
    baseModel = 'Salesforce/codet5-small'
    pretrained_model_path = "core/CMIF/utils/t5_classification_final_ep2.mdl"
    logger.info('Loading tokenizer: %s, Pretrained Model: %s', baseModel, pretrained_model_path)

    tokenizer = RobertaTokenizer.from_pretrained(baseModel)
    model1 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)

    input_path = os.path.join(DATA_DIR, "input.jsonl")
    data = load_jsonl(input_path)
    model1 = model1.to("cpu")
    index_c = 0

    predictions = []
    for test_element in data:
        test_source = test_element['source']
        inputs = tokenizer(test_source, return_tensors='pt').input_ids
        generated_ids = model1.generate(inputs.to("cpu"), num_beams=5, max_length=300, num_return_sequences=1)
        for i, beam_output in enumerate(generated_ids):
            fix = tokenizer.decode(beam_output, skip_special_tokens=True)
            if fix.lower() == "consistent":
                predictions.append(
                    {"source": test_source}
                )
        index_c += 1

    output_path = os.path.join(DEST_DIR, "output.jsonl")
    dump_jsonl(predictions, output_path)
