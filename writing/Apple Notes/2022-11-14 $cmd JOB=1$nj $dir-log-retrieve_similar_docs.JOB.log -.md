$cmd JOB=1:$nj $dir/log/retrieve_similar_docs.JOB.log \
    steps/cleanup/internal/retrieve_similar_docs.py \
      --query-tfidf=$dir/query_docs/split$nj/query_tf_idf.JOB.ark.txt \
      --source-text-id2tfidf=$dir/docs/source2tf_idf.scp \
      --source-text-id2doc-ids=$dir/docs/text2doc \
      --query-id2source-text-id=$dir/new2orig_utt \
      --num-neighbors-to-search=1 \
      --neighbor-tfidf-threshold=0.5 \
      --relevant-docs=$dir/query_docs/split$nj/relevant_docs.JOB.txt