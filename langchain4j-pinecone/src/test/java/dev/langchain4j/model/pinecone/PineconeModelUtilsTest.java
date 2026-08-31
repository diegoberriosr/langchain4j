package dev.langchain4j.model.pinecone;

import static dev.langchain4j.model.embedding.request.EmbeddingInputType.DOCUMENT;
import static dev.langchain4j.model.pinecone.PineconeModelUtils.buildModelParameters;
import static dev.langchain4j.model.pinecone.PineconeModelUtils.isSparseEmbeddingResponse;
import static dev.langchain4j.model.pinecone.PineconeModelUtils.toEmbedding;
import static org.assertj.core.api.Assertions.assertThat;

import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.openapitools.inference.client.model.Embedding;
import org.openapitools.inference.client.model.EmbeddingsList;

class PineconeModelUtilsTest {

    @ParameterizedTest
    @MethodSource("sparseEmbeddingResponses")
    void isSparseEmbeddingResponse_should_detect_sparse_embeddings(EmbeddingsList embeddingList) {

        // given - when - then
        assertThat(isSparseEmbeddingResponse(embeddingList)).isTrue();
    }

    @Test
    void isSparseEmbeddingResponse_should_return_false_for_dense_embeddings() {

        // given
        EmbeddingsList embeddingList =
                embeddingsList(new Embedding().values(List.of(BigDecimal.valueOf(0.5), BigDecimal.valueOf(0.7))));

        // when - then
        assertThat(isSparseEmbeddingResponse(embeddingList)).isFalse();
    }

    @Test
    void toEmbedding_should_convert_values_to_a_float_vector() {

        // given
        Embedding sdkEmbedding = new Embedding().values(List.of(BigDecimal.valueOf(0.5), BigDecimal.valueOf(0.7)));

        // when
        dev.langchain4j.data.embedding.Embedding embedding = toEmbedding(sdkEmbedding);

        // then
        assertThat(embedding.vector()).containsExactly(0.5f, 0.7f);
    }

    @Test
    void buildModelParameters_should_default_input_type_if_NOT_present_in_parameters() {

        // given - when
        Map<String, Object> modelParameters = buildModelParameters(EmbeddingRequestParameters.EMPTY);

        // then
        assertThat(modelParameters).containsExactly(Map.entry("input_type", "query"));
    }

    @Test
    void buildModelParameters_should_append_input_type_if_present_in_parameters() {

        // given
        EmbeddingRequestParameters parameters =
                EmbeddingRequestParameters.builder().inputType(DOCUMENT).build();

        // when
        Map<String, Object> modelParameters = buildModelParameters(parameters);

        // then
        assertThat(modelParameters).containsExactly(Map.entry("input_type", "passage"));
    }

    static Stream<EmbeddingsList> sparseEmbeddingResponses() {
        return Stream.of(
                embeddingsList(new Embedding().putAdditionalProperty("sparse_indices", List.of(1, 2))),
                embeddingsList(new Embedding().putAdditionalProperty("sparse_values", List.of(0.5, 0.7))),
                embeddingsList(new Embedding()
                        .putAdditionalProperty("sparse_indices", List.of(1, 2))
                        .putAdditionalProperty("sparse_values", List.of(0.5, 0.7))));
    }

    private static EmbeddingsList embeddingsList(Embedding... embeddings) {
        return new EmbeddingsList().data(List.of(embeddings));
    }
}
