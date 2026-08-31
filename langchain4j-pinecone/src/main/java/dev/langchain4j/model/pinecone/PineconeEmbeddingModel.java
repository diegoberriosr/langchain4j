package dev.langchain4j.model.pinecone;

import static dev.langchain4j.internal.Exceptions.illegalArgument;
import static dev.langchain4j.internal.RetryUtils.withRetryMappingExceptions;
import static dev.langchain4j.internal.Utils.copy;
import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.internal.Utils.isNotNullOrBlank;
import static dev.langchain4j.internal.ValidationUtils.ensureNotBlank;
import static dev.langchain4j.model.embedding.request.EmbeddingRequestParameters.INPUT_TYPE;
import static dev.langchain4j.model.embedding.request.EmbeddingRequestParameters.MODEL_NAME;
import static dev.langchain4j.model.pinecone.PineconeModelUtils.buildModelParameters;
import static dev.langchain4j.model.pinecone.PineconeModelUtils.isSparseEmbeddingResponse;
import static java.util.Arrays.asList;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.listener.EmbeddingModelListener;
import dev.langchain4j.model.embedding.request.EmbeddingInput;
import dev.langchain4j.model.embedding.request.EmbeddingParameter;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import dev.langchain4j.model.output.TokenUsage;
import io.pinecone.clients.Inference;
import io.pinecone.clients.Pinecone;
import io.pinecone.configs.PineconeConfig;
import java.util.List;
import java.util.Set;
import org.openapitools.inference.client.model.EmbeddingsList;

/**
 * An implementation of {@link EmbeddingModel} that uses Pinecone's inference API's {@code /embed} endpoint.
 *
 * @see <a href="https://docs.pinecone.io/reference/api/2026-04/inference/generate-embeddings">Pinecone's API embedding endpoint docs</a>
 */
public class PineconeEmbeddingModel implements EmbeddingModel {

    private final Inference pineconeInferenceClient;
    private final String model;

    private final List<EmbeddingModelListener> listeners;
    private final Integer maxRetries;

    private PineconeEmbeddingModel(Builder builder) {
        if (builder.pinecone != null) {
            this.pineconeInferenceClient = builder.pinecone.getInferenceClient();
        } else if (isNotNullOrBlank(builder.apiKey)) {
            this.pineconeInferenceClient = new Inference(new PineconeConfig(builder.apiKey));
        } else {
            throw illegalArgument("Either use a custom Pinecone client or provide an API key.");
        }

        this.model = ensureNotBlank(builder.model, "Model name");

        this.listeners = copy(builder.listeners);
        this.maxRetries = getOrDefault(builder.maxRetries, 3);
    }

    @Override
    public EmbeddingResponse doEmbed(EmbeddingRequest request) {
        List<String> inputs =
                request.inputs().stream().map(EmbeddingInput::text).toList();

        EmbeddingRequestParameters parameters = request.parameters();

        EmbeddingsList response = withRetryMappingExceptions(
                () -> pineconeInferenceClient.embed(
                        getOrDefault(parameters.parameter(MODEL_NAME), model),
                        buildModelParameters(parameters),
                        inputs),
                maxRetries);

        if (isSparseEmbeddingResponse(response)) {
            throw new UnsupportedFeatureException("Sparse vectors are not supported");
        }

        List<Embedding> embeddings =
                response.getData().stream().map(PineconeModelUtils::toEmbedding).toList();

        return EmbeddingResponse.builder()
                .embeddings(embeddings)
                .modelName(response.getModel())
                .tokenUsage(new TokenUsage(response.getUsage().getTotalTokens()))
                .build();
    }

    @Override
    public String modelName() {
        return model;
    }

    @Override
    public List<EmbeddingModelListener> listeners() {
        return listeners;
    }

    @Override
    public Set<EmbeddingParameter<?>> supportedParameters() {
        return Set.of(INPUT_TYPE, MODEL_NAME);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        private String apiKey;
        private String model;
        private List<EmbeddingModelListener> listeners;
        private Integer maxRetries;
        private Pinecone pinecone;

        /**
         * Pinecone's API key.
         */
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        /**
         * The model used to generate the embeddings.
         * @see <a href="https://docs.pinecone.io/guides/index-data/create-an-index#embedding-models">Pinecone's supported embedding models</a>
         */
        public Builder model(String model) {
            this.model = model;
            return this;
        }

        /**
         * Plug in custom listeners to add observability to the embedding model.
         * @see <a href="https://docs.langchain4j.dev/tutorials/observability#embeddingmodel-listener">Observability for embedding models</a>
         */
        public Builder listeners(List<EmbeddingModelListener> listeners) {
            this.listeners = listeners;
            return this;
        }

        public Builder listeners(EmbeddingModelListener... listeners) {
            return listeners(asList(listeners));
        }

        /**
         * Maximum number of retries for failed requests. Defaults to {@code 3}.
         */
        public Builder maxRetries(Integer maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }

        /**
         * Use a custom Pinecone client. Its configuration (API key, custom OkHttp client, proxy)
         * is reused for the inference calls.
         */
        public Builder pinecone(Pinecone pinecone) {
            this.pinecone = pinecone;
            return this;
        }

        public PineconeEmbeddingModel build() {
            return new PineconeEmbeddingModel(this);
        }
    }
}
