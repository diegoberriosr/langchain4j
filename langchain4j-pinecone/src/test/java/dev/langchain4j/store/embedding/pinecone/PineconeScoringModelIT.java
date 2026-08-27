package dev.langchain4j.store.embedding.pinecone;

import static dev.langchain4j.data.segment.TextSegment.textSegment;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.scoring.ScoringModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EmptySource;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;

@EnabledIfEnvironmentVariable(named = "PINECONE_API_KEY", matches = ".+")
class PineconeScoringModelIT {

    private static final String QUERY = "What is a dog?";

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"", " "})
    void should_throw_exception_during_instantiation_if_api_key_is_not_defined(String illegalApiKey) {

        // given
        PineconeScoringModel.Builder builder = PineconeScoringModel.builder().apiKey(illegalApiKey);

        // when - then
        assertThatThrownBy(builder::build).isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"", " "})
    void should_throw_exception_during_instantiation_if_model_is_NOT_defined(String illegalModelName) {

        // given
        PineconeScoringModel.Builder builder = PineconeScoringModel.builder()
                        .apiKey(System.getenv("PINECONE_API_KEY"))
                        .modelName(illegalModelName);

        // when - then
        assertThatThrownBy(builder::build).isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"", " "})
    void should_return_empty_response_if_query_is_invalid(String invalidQuery) {

        // given
        ScoringModel scoringModel = PineconeScoringModel.builder()
                .apiKey(System.getenv("PINECONE_API_KEY"))
                .modelName("pinecone-rerank-v0")
                .build();

        List<TextSegment> textSegments = List.of(textSegment("Random content"));

        // when
        List<Double> scores = scoringModel.scoreAll(textSegments, invalidQuery).content();

        // then
        assertThat(scores).isEmpty();
    }

    @ParameterizedTest
    @NullSource
    @EmptySource
    void should_return_empty_response_if_text_segments_are_invalid(List<TextSegment> invalidTextSegments) {

        // given
        ScoringModel scoringModel = PineconeScoringModel.builder()
                .apiKey(System.getenv("PINECONE_API_KEY"))
                .modelName("pinecone-rerank-v0")
                .build();

        // when
        List<Double> scores = scoringModel.scoreAll(invalidTextSegments, QUERY).content();

        // then
        assertThat(scores).isEmpty();
    }

    @Test
    void should_score_single_text() {

        // given
        ScoringModel scoringModel = PineconeScoringModel.builder()
                .apiKey(System.getenv("PINECONE_API_KEY"))
                .modelName("pinecone-rerank-v0")
                .build();

        String document = "Dogs are domesticated descendant of wolves";

        // when
        Double score = scoringModel.score(document, QUERY).content();

        // then
        assertThat(score)
                .isNotNull()
                .isGreaterThan(0);
    }

    @Test
    void should_score_multiple_documents() {

        // given
        ScoringModel scoringModel = PineconeScoringModel.builder()
                .apiKey(System.getenv("PINECONE_API_KEY"))
                .modelName("pinecone-rerank-v0")
                .build();

        List<TextSegment> documents = List.of(
                TextSegment.from("A dog is a descendant of wolves"),
                TextSegment.from("Hünde sind haustieren"),
                TextSegment.from("狗是宠物"));

        // when
        List<Double> scores = scoringModel.scoreAll(documents, QUERY).content();

        // then
        assertThat(scores)
                .hasSize(3)
                .allMatch(score -> score >= 0);
    }

    @Test
    void should_limit_score_size_if_top_N_is_defined() {

        // given
        ScoringModel scoringModel = PineconeScoringModel.builder()
                .apiKey(System.getenv("PINECONE_API_KEY"))
                .modelName("pinecone-rerank-v0")
                .topN(1)
                .build();

        List<TextSegment> documents = List.of(
                TextSegment.from("A dog is a descendant of wolves"),
                TextSegment.from("Hünde sind haustieren"),
                TextSegment.from("狗是宠物"));

        // when
        List<Double> scores = scoringModel.scoreAll(documents, QUERY).content();

        // then
        assertThat(scores)
                .hasSize(1)
                .allMatch(score -> score >= 0);
    }
}
