//Term Frequency/Inverse Document Frequency Calculator

class TFIDF {

    val tfIdfVectors: MutableList<MutableMap<String, Double>> = mutableListOf()
    private val corpus: MutableSet<String> = mutableSetOf("%UNCATEGORIZED%")

    private fun buildCorpus(documents: List<List<String>>) {
        for (document in documents) {
            for (term in document) {
                corpus.add(term)
            }
        }
    }

    private fun tf(document: List<String>): MutableMap<String, Double> {
        val docTf: MutableMap<String, Double> = mutableMapOf()
        for (term in document) {
            docTf[term] =  docTf.getOrDefault(term, 0.0) + 1
        }
        return docTf
    }

    private fun tfs(documents: List<List<String>>): List<MutableMap<String, Double>> {

        val tfs: List<MutableMap<String, Double>> = listOf()
        for (document in documents) {
            tfs.plus(tf(document))
        }
        return tfs
    }

    private fun tfs(document: List<String>) {

    }

    private fun idf(documents: List<List<String>>): MutableMap<String, Double> {
        //calculate document frequency
        val df: MutableMap<String, Double> = mutableMapOf()
        //n documents. Assume there exists one document containing every word once.
        var n = 1
        for (document in documents) {
            n++
            for (term in document) {
                df[term] = df.getOrDefault(term, 1.0) + 1
            }
            //calculate inverse document frequency
        }
        val idf: MutableMap<String, Double> = mutableMapOf()
        for ((term, docFrequency) in df) {
            idf[term] = Math.log(n / docFrequency) + 1
        }
        return idf
    }

    //document level
    private fun tfIdf(tf: MutableMap<String, Double>, idf: MutableMap<String, Double>): MutableMap<String, Double> {
        val tfIdf: MutableMap<String, Double> = mutableMapOf()
        for ((term, inverseFrequency) in idf) {
            //raises issue: how to guarantee tf & idf computed for each term?
            //what is a reasonable default for idf
            tfIdf[term] = (tf[term]?: 1.0) * inverseFrequency
        }
        //normalize
        return tfIdf
    }

    //calculate TF-IDF for a number of documents
    //use this to train on a processed Corpus
    fun fitDocuments(documents: List<List<String>>) {
        val tfs = tfs(documents)
        val idf = idf(documents)
        for (tf in tfs) {
            tfIdfVectors.add(tfIdf(tf, idf))
        }
        buildCorpus(documents)
    }

    fun addDocument(document: List<String>) {
        val cleanedDoc: MutableList<String> = mutableListOf()
        for (term in document) {
            if (corpus.contains(term)) {
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
