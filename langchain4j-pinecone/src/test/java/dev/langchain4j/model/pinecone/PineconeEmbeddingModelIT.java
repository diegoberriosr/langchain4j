package dev.langchain4j.model.pinecone;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.common.AbstractEmbeddingModelIT;
import dev.langchain4j.model.embedding.listener.EmbeddingModelListener;
import io.pinecone.clients.Pinecone;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

@EnabledIfEnvironmentVariable(named = "PINECONE_API_KEY", matches = ".+")
class PineconeEmbeddingModelIT extends AbstractEmbeddingModelIT {

    private static final String API_KEY = System.getenv("PINECONE_API_KEY");
    private static final String MODEL = "llama-text-embed-v2";
    private static final String SPARSE_MODEL_NAME = "pinecone-sparse-english-v0";

    @Override
    protected List<EmbeddingModel> models() {
        return List.of(
                PineconeEmbeddingModel.builder().apiKey(API_KEY).model(MODEL).build(),
                PineconeEmbeddingModel.builder()
                        .pinecone(new Pinecone.Builder(API_KEY).build())
                        .model(MODEL)
                        .build());
    }

    @Override
    protected EmbeddingModel modelWith(EmbeddingModelListener listener) {
        return PineconeEmbeddingModel.builder()
                .apiKey(API_KEY)
                .model(MODEL)
                .listeners(listener)
                .build();
    }

    @Override
    protected EmbeddingModel failingModelWith(EmbeddingModelListener listener) {
        return PineconeEmbeddingModel.builder()
                .apiKey("banana")
                .model(MODEL)
                .maxRetries(0)
                .listeners(listener)
                .build();
    }

    @Override
    protected boolean supportsImageInput() {
        return false;
    }

    @Override
    protected boolean supportsDimensionsParameter() {
        return false;
    }

    @Test
    void should_NOT_embed_single_text_for_sparse_models() {

        // given
        PineconeEmbeddingModel model = PineconeEmbeddingModel.builder()
                .apiKey(API_KEY)
                .model(SPARSE_MODEL_NAME)
                .build();

        // when - then
        assertThatThrownBy(() -> model.embed("The quick brown fox jumps over the lazy dog"))
                .isExactlyInstanceOf(UnsupportedFeatureException.class);
    }

    @Test
    void should_NOT_embed_multiple_text_segments_for_sparse_models() {

        // given
        PineconeEmbeddingModel model = PineconeEmbeddingModel.builder()
                .apiKey(API_KEY)
                .model(SPARSE_MODEL_NAME)
                .build();

        List<TextSegment> segments = List.of(
                TextSegment.from("Peter Piper picked a peck of pickled peppers"),
                TextSegment.from("Tres tristes tigres comían trigo en un trigal"),
                TextSegment.from("Fischers Fritze fischt frische Fische"));

        // when - then
        assertThatThrownBy(() -> model.embedAll(segments)).isExactlyInstanceOf(UnsupportedFeatureException.class);
    }
}
