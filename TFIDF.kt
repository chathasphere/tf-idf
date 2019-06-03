/**
 * Implementation of the Term Frequency/Inverse Document Frequency algorithm. Tokenizes a corpus
 * (collection of documents) by assigning the TFIDF statistic to each word, measuring its overall importance to the
 * corpus. A count of how often a word appears in its document (TF) is multiplied by a log-scaled inverse count of
 * how often the word appears in the corpus (IDF). This measures the "specificity" of a word. For instance, in a
 * typical English corpus, "the" and "a" will have a high TF, but their significance will be offset by a low IDF.
 *
 * The intuition of IDF follows Zipf's law, the observation that the distribution of terms in a natural language corpus
 * follows a "power law": a word's frequency ranking is inversely proportional to its frequency. A logarithm helps
 * smooth the distribution.
 *
 * This class "trains" on an initial corpus, a list of documents. Each document should be a list of words. Further
 * documents may be added to the model with a lower computational cost, with the caveat that new words will be
 * ignored (cast to "%UNCATEGORIZED%").
 *
 * @property tfIdfVectors a list (one per document in corpus) of maps assigning TF-IDF to a document's words
 * @property knownWords a set of all recognized words in the corpus.
 * @property df a mapping of words to the number of documents in which they appear
 */


class TFIDF {

    val tfIdfVectors: MutableList<MutableMap<String, Double>> = mutableListOf()
    private val knownWords: MutableSet<String> = mutableSetOf("%UNCATEGORIZED%")
    private val df: MutableMap<String, Double> = mutableMapOf()

    // Adds words from the training corpus to the known words set
    private fun findKnownWords(documents: List<List<String>>) {
        for (document in documents) {
            for (term in document) {
                knownWords.add(term)
            }
        }
    }

    // Calculates the frequency of a terms in a document and returns a map
    private fun tf(document: List<String>): MutableMap<String, Double> {
        val docTf: MutableMap<String, Double> = mutableMapOf()
        for (term in document) {
            docTf[term] =  docTf.getOrDefault(term, 0.0) + 1
        }
        return docTf
    }

    // Calls tf for a list of documents. Returns a list of  TF mappings
    private fun tfs(documents: List<List<String>>): List<MutableMap<String, Double>> {

        val tfs: List<MutableMap<String, Double>> = listOf()
        for (document in documents) {
            tfs.plus(tf(document))
        }
        return tfs
    }

    // Given a corpus, returns a map of words to their inverse document frequency
    private fun idf(documents: List<List<String>>): MutableMap<String, Double> {
        //First, calculates "document frequency" of words in the
        //n documents. Assume there exists one "master" document containing every word once.
        var n = 1
        for (document in documents) {
            n++
            for (term in document) {
                df[term] = df.getOrDefault(term, 1.0) + 1
            }
        }
        //Next, calculate inverse document frequency
        val idf: MutableMap<String, Double> = mutableMapOf()
        for ((term, docFrequency) in df) {
            idf[term] = Math.log(n / docFrequency) + 1
        }
        return idf
    }

    //Calculates TFIDF for words in a document
    private fun tfIdf(tf: MutableMap<String, Double>, idf: MutableMap<String, Double>): MutableMap<String, Double> {
        val tfIdf: MutableMap<String, Double> = mutableMapOf()
        for ((term, inverseFrequency) in idf) {
            tfIdf[term] = (tf[term]?: 1.0) * inverseFrequency
        }
        return tfIdf
    }

    //Calculates TF-IDF for a training corpus (list of documents, each a list of words)
    fun fitDocuments(documents: List<List<String>>) {
        val tfs = tfs(documents)
        val idf = idf(documents)
        for (tf in tfs) {
            tfIdfVectors.add(tfIdf(tf, idf))
        }
        findKnownWords(documents)
    }

    //Calculate TF-IDF for a new document and add to saved TFIDF vectors
    fun addDocument(document: List<String>) {
        val cleanedDoc: MutableList<String> = mutableListOf()
        for (term in document) {
            if (knownWords.contains(term)) {
                cleanedDoc.add(term)
            } else {
                cleanedDoc.add("%UNCATEGORIZED%")
            }
        }
        val tf = tf(cleanedDoc)
        val idf = idf(listOf(cleanedDoc))
        tfIdfVectors.add(tfIdf(tf, idf))
    }

}
