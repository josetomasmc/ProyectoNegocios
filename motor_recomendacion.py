{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d1c51515",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\Jose\\anaconda3\\envs\\ProyectoNegocios\\Lib\\site-packages\\tqdm\\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html\n",
      "  from .autonotebook import tqdm as notebook_tqdm\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sentence_transformers import SentenceTransformer\n",
    "\n",
    "class RecommenderEngine:\n",
    "    \"\"\"Motor simple que carga embeddings y busca vecinos por similitud coseno.\"\"\"\n",
    "    def __init__(self, model_name: str, embedding_file: str, index_file: str):\n",
    "        if not os.path.exists(embedding_file) or not os.path.exists(index_file):\n",
    "            raise FileNotFoundError(f\"Faltan archivos: {embedding_file} o {index_file}\")\n",
    "\n",
    "        self.model = SentenceTransformer(model_name)\n",
    "        self.embeddings = np.load(embedding_file)\n",
    "        self.index = pd.read_csv(index_file)\n",
    "\n",
    "        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)\n",
    "        norms[norms == 0] = 1.0\n",
    "        self.norm_embeddings = self.embeddings / norms\n",
    "\n",
    "    def find_similar(self, query_text: str, top_n: int = 5) -> pd.DataFrame:\n",
    "        q_vec = self.model.encode([query_text])\n",
    "        q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10)\n",
    "        sims = np.dot(self.norm_embeddings, q_norm[0])\n",
    "        top_idx = np.argsort(-sims)[:top_n]\n",
    "\n",
    "        results = self.index.iloc[top_idx].copy().reset_index(drop=True)\n",
    "        results['score'] = sims[top_idx]\n",
    "        return results"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ProyectoNegocios",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
