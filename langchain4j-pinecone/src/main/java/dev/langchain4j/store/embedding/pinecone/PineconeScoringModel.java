package dev.langchain4j.store.embedding.pinecone;

import static dev.langchain4j.internal.RetryUtils.withRetryMappingExceptions;
import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.Utils.isNullOrBlank;
import static dev.langchain4j.internal.Utils.isNullOrEmpty;
import static dev.langchain4j.internal.ValidationUtils.ensureNotBlank;
import static java.util.Collections.emptyList;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.scoring.ScoringModel;
import io.pinecone.clients.Pinecone;
import org.openapitools.inference.client.model.RerankResult;

/**
 * A {@link ScoringModel} implementation that uses {@code Pinecone}'s reranking API.
 *
 * @see <a href="https://docs.pinecone.io/reference/api/2025-10/inference/rerank">Rerank endpoint official documentation</a>
 */
public class PineconeScoringModel implements ScoringModel {

    private static final List<String> RERANKING_FIELD = List.of("text");

    private final Pinecone pinecone;
    private final String model;
    private final Integer topN;
    private final Integer maxRetries;

    private PineconeScoringModel(Builder builder) {
        this.pinecone = new Pinecone.Builder(ensureNotBlank(builder.apiKey, "API key"))
                .build();

        this.model = ensureNotBlank(builder.modelName, "Model name");
        this.topN = builder.topN;
        this.maxRetries = getOrDefault(builder.maxRetries, 3);
    }

    @Override
    public Response<List<Double>> scoreAll(List<TextSegment> segments, String query) {
        if (isNullOrBlank(query) || isNullOrEmpty(segments)) {
            return Response.from(emptyList());
        }

        var documents = segments.stream()
                .map(segment -> Map.of("text", segment.text()))
                .toList();

        RerankResult rerankResult = withRetryMappingExceptions(
                () -> pinecone.getInferenceClient().rerank(
                        model,
                        query,
                        documents,
                        RERANKING_FIELD,
                        getOrDefault(topN, documents.size()),
                        false,
                        null),
                maxRetries);

        return Response.from(rerankResult.getData()
                .stream()
                .map(d -> d.getScore().doubleValue())
                .toList());
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private String apiKey;
        private String modelName;
        private Integer topN;
        private Integer maxRetries;

        /**
         * Sets the {@code Pinecone} API key
         */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * The model to use for reranking.
         */
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * The number of results to return sorted by relevance, defaults to the number of {@link TextSegment}s
         * you send in the request.
         */
        public Builder topN(Integer topN) {
            this.topN = topN;
            return this;
        }

        /**
         * How many retries the client is allowed to execute the request after a failure Defaults to {@code 3}>
         */
        public Builder maxRetries(Integer maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        public PineconeScoringModel build() {
            return new PineconeScoringModel(this);
        }
    }
}
