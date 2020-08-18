import numpy as np


class LDA:

    def __init__(self, docs, _topic_num, _alpha=0.1, _beta=0.01, _max_iter=10):
        self.docs = docs
        self.vocabulary = self.get_unique_words()

        self._topic_num = _topic_num
        self.N_D = docs.__len__()  # number of documents
        self.N_W = self.vocabulary.__len__()  # number of words

        self._alpha_origin = self._alpha
        self._alpha = self._init_alpha(_alpha)  # document topic distribution
        self._beta_origin = self._beta
        self._beta = self._init_beta(_beta)  # word topic distribution
        self._zeta = self._init_zeta()  # word topic assignment
        # doc to word maps where every element is the number of the word in the doc
        self.X = self._vectorize_docs()

        self._max_iter = _max_iter

    def get_unique_words(self):
        concatenated_list = []
        for doc in self.docs:
            concatenated_list += doc
        return list(set(concatenated_list))

    def _init_alpha(self, _alpha):
        self._alpha = np.zeros([self.N_D, self._topic_num])
        for i in range(self.N_D):
            self._alpha[i] = np.random.dirichlet(
                _alpha*np.ones(self._topic_num))

        return self._alpha

    def _init_beta(self, _beta):
        self._beta = np.zeros([self._topic_num, self.N_W])
        for i in range(self.N_D):
            self._beta[i] = np.random.dirichlet(
                _beta*np.ones(self.N_W))

        return self._beta

    def _init_zeta(self):
        self._zeta = np.zeros([self.N_D, self.N_W])
        for i in range(self.N_D):
            for j in range(self.N_W):
                self._zeta[i, j] = np.random.randint(self._topic_num)

        return self._zeta

    def _vectorize_docs(self):
        self.X = np.zeros([self.N_D, self.N_W])
        for i in range(self.N_D):
            for j in range(self.N_W):
                self.X[i, j] = len(
                    [1 for w in self.docs[i] if w == self.vocabulary[j]])

        return self.X

    def _lda_single_iter(self):
        for i in range(self.N_D):
            for v in range(self.N_W):
                p_iv = np.exp(
                    np.log(self._alpha[i]) + np.log(self._beta[:, self.X[i, v]]))
                p_iv /= np.sum(p_iv)
                self._zeta[i, v] = np.random.multinomial(1, p_iv).argmax()

        for i in range(self.N_D):
            m = np.zeros(self._topic_num)

            for k in range(self._topic_num):
                m[k] = np.sum(self._zeta[i] == k)

            self._alpha[i, :] = np.random.dirichlet(m + self._alpha_origin)

        for k in range(self._topic_num):
            n = np.zeros(self.N_W)

            for v in range(self.N_W):
                for i in range(self.N_D):
                    for l in range(self.N_W):
                        n[v] += (self.X[i, l] == v) and (self._zeta[i, l] == k)

            self._beta[k, :] = np.random.dirichlet(self._beta_origin + n)

    def fit(self):
        for i in range(self._max_iter):
            self._lda_single_iter()
