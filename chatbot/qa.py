from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
import os
import argparse
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def parse_arguments():
    parser = argparse.ArgumentParser(description='GPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


def retrieval_qa(db) :
    model_type = os.environ.get('MODEL_TYPE')
    model_path = os.environ.get('MODEL_PATH')
    model_n_ctx = os.environ.get('MODEL_N_CTX')
    model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
    target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
    args = parse_arguments()
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)