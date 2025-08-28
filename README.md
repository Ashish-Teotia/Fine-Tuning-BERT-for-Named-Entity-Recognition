# Fine-Tuning-BERT-for-Named-Entity-Recognition

This project focuses on fine-tuning a pre-trained BERT model (bert-base-cased) for the task of Named Entity Recognition (NER) using the CoNLL-2003 dataset. The implementation utilizes the powerful HuggingFace transformers and datasets libraries to handle data loading, model architecture, and training workflows. The goal is to accurately identify named entities such as persons, organizations, locations, and miscellaneous entities within text.

To prepare the dataset, the tokens were preprocessed and aligned with their corresponding NER labels using BertTokenizerFast. Special attention was given to handling subword tokens properly. A custom label alignment function ensures that only valid token positions contribute to the training loss by ignoring special tokens and padding. The CoNLL-2003 dataset, which includes four types of entity labels (PER, ORG, LOC, MISC), was directly loaded using HuggingFace's load_dataset function.

The model was fine-tuned using AutoModelForTokenClassification, with 9 output labels corresponding to the entity tag set. Training was conducted using HuggingFace's Trainer API, which simplifies the training loop and integrates seamlessly with evaluation metrics. Dynamic padding was managed using DataCollatorForTokenClassification, and the model was evaluated using the seqeval library. The fine-tuned model achieved a strong performance with an F1 score of 93%, along with high precision, recall, and accuracy metrics.

Once trained, the model can be easily used for inference with HuggingFace's pipeline interface. For example, you can input sentences such as "Barack Obama was born in Hawaii and worked at the White House.", and the pipeline will return the recognized named entities. All training outputs, including model weights and logs, are saved in the ./bert-ner/ directory.

To run this project, you can install the required dependencies using pip install transformers datasets seqeval. Future improvements may include support for custom datasets, model optimization via hyperparameter tuning, and exporting the model for deployment using ONNX or TensorFlow Serving.
