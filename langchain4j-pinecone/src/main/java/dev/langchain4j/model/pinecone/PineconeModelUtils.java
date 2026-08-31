package dev.langchain4j.model.pinecone;

import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.model.embedding.request.EmbeddingInputType.QUERY;
import static dev.langchain4j.model.embedding.request.EmbeddingRequestParameters.INPUT_TYPE;

import dev.langchain4j.Internal;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.request.EmbeddingInputType;
import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import org.openapitools.inference.client.model.EmbeddingsList;

/**
 * Helper for {@link PineconeEmbeddingModel} request and response conversion
 */
@Internal
class PineconeModelUtils {

    private PineconeModelUtils() throws InstantiationException {
        throw new InstantiationException("can't instantiate this class");
    }

    public static boolean isSparseEmbeddingResponse(EmbeddingsList embeddings) {
        return embeddings.getData().stream()
                .anyMatch(embedding -> embedding.getAdditionalProperty("sparse_indices") != null
                        || embedding.getAdditionalProperty("sparse_values") != null);
    }

    public static Embedding toEmbedding(org.openapitools.inference.client.model.Embedding embedding) {
        List<BigDecimal> values = embedding.getValues();
        float[] vector = new float[values.size()];
        for (int i = 0; i < vector.length; i++) {
            vector[i] = values.get(i).floatValue();
        }

        return Embedding.from(vector);
    }

    public static Map<String, Object> buildModelParameters(EmbeddingRequestParameters parameters) {
        // `input_type` is a required parameter for _every_ embedding request, but the Pinecone 3.10 SDK
        // does not automatically append a default value if not defined!
        // See https://docs.pinecone.io/guides/index-data/create-an-index#embedding-models
        EmbeddingInputType inputType = getOrDefault(parameters.parameter(INPUT_TYPE), QUERY);
        return Map.of("input_type", mapInputType(inputType));
    }

    private static String mapInputType(EmbeddingInputType type) {
        return switch (type) {
            case QUERY -> "query";
            case DOCUMENT -> "passage";
        };
    }
}
