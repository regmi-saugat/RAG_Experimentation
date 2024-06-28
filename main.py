from src.DataLoading import DatasetLoading
from src.RAGModels import RagModel

import argparse
import time
import torch

from loguru import logger


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device} as the device")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2b-it")
    parser.add_argument("--encoding_model_name", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--use_quantization", type=bool, default=True)
    args = parser.parse_args()

    embedding_model_name = args.encoding_model_name
    model_name = args.model_name
    use_quantization = args.use_quantization

    logger.info(
        "Encoding Model: {} \nLLM model: {} \nQuantisation is set to: {}".format(
            embedding_model_name, model_name, use_quantization
        )
    )
    start_time = time.time()
    dataset_load = DatasetLoading(
        df_path="text_chunks_and_embeddings_df.csv", device=device
    )
    dataset_load.save_embeddings()
    rag_model = RagModel(
        model_id=model_name,
        embedding_model_name=embedding_model_name,
        use_quantization=use_quantization,
        device=device,
    )
    end_time = time.time()

    logger.info(f"Runtime: {round(end_time - start_time, 2)} seconds")

    while True:
        start_time = time.time()
        print("Enter your query")
        query = str(input())
        query_embeddings = rag_model.embedding_model.encode(query)
        _, samples = dataset_load.retun_faiss_index(query_embeddings)
        prompt = rag_model.create_prompt(samples["sentence_chunk"], query)

        llm_response = rag_model.generate(prompt)

        logger.info(llm_response)
        end_time = time.time()
        logger.info(f"Runtime: {round(end_time - start_time, 2)} seconds")
        return llm_response


if __name__ == "__main__":
    main()
