import os


class USEEmbedder:
    def __init__(self, force_cpu=False):
        if force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.environ["HOME"], ".cache/tfhub")
        import tensorflow_hub as hub
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(module_url)

    def embed_many(self, sentences):
        return self.model(sentences).numpy()

    def embed_one(self, sentence):
        return self.embed_many([sentence])


use_embedder = USEEmbedder(force_cpu=True)
