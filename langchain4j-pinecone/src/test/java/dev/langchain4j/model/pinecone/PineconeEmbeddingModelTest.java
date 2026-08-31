package dev.langchain4j.model.pinecone;

import static org.assertj.core.api.Assertions.assertThatNoException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import io.pinecone.clients.Pinecone;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;

class PineconeEmbeddingModelTest {

    private static final String DUMMY_API_KEY = "dummy-api-key";
    private static final String MODEL = "llama-text-embed-v2";

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"", " "})
    void should_throw_during_instantiation_if_invalid_api_key(String invalidApiKey) {

        // given
        PineconeEmbeddingModel.Builder builder =
                PineconeEmbeddingModel.builder().apiKey(invalidApiKey).model(MODEL);

        // when
        assertThatThrownBy(builder::build).isExactlyInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"", " "})
    void should_throw_during_instantiation_if_invalid_model_name(String invalidModelName) {

        // given
        PineconeEmbeddingModel.Builder builder =
                PineconeEmbeddingModel.builder().apiKey(DUMMY_API_KEY).model(invalidModelName);

        // when
        assertThatThrownBy(builder::build).isExactlyInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void should_NOT_throw_during_instantiation_if_custom_client_and_no_api_key() {

        // given
        Pinecone customClient = new Pinecone.Builder(DUMMY_API_KEY).build();

        PineconeEmbeddingModel.Builder builder =
                PineconeEmbeddingModel.builder().pinecone(customClient).model(MODEL);

        // when - then
        assertThatNoException().isThrownBy(builder::build);
    }

    @Test
    void should_throw_during_instantiation_if_neither_api_key_nor_custom_client() {

        // given
        PineconeEmbeddingModel.Builder builder =
                PineconeEmbeddingModel.builder().model(MODEL);

        // when - then
        assertThatThrownBy(builder::build).isExactlyInstanceOf(IllegalArgumentException.class);
    }
}
