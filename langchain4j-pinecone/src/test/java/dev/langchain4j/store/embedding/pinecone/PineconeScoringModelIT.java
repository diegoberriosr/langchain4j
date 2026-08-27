package dev.langchain4j.store.embedding.pinecone;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.scoring.ScoringModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;
import java.util.List;

@EnabledIfEnvironmentVariable(named = "PINECONE_API_KEY", matches = ".+")
class PineconeScoringModelIT {

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"", " "})
    void should_throw_exception_during_instantiation_if_api_key_is_not_defined(String illegalApiKey) {

        // given - when - then
        assertThatThrownBy(() -> PineconeScoringModel.builder().apiKey(illegalApiKey).build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"", " "})
    void should_throw_exception_during_instantiation_if_model_is_not_defined(String illegalModelName) {

        // given - when - then
        assertThatThrownBy(() -> PineconeScoringModel.builder().apiKey(System.getenv("PINECONE_API_KEY")).modelName(illegalModelName).build())
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void should_score_single_text() {

        // given
        ScoringModel scoringModel = PineconeScoringModel.builder()
                .apiKey(System.getenv("PINECONE_API_KEY"))
                .modelName("pinecone-rerank-v0")
                .build();

        String query = "What is a dog?";
        String document = "Dogs are domesticated descendant of wolves";

        // when
        Double score = scoringModel.score(document, query).content();

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

        String query = "What is a dog?";
        List<TextSegment> documents = List.of(
                TextSegment.from("A dog is a descendant of wolves"),
                TextSegment.from("Hünde sind haustieren"),
                TextSegment.from("Cinco fases de un pene"));

        // when
        List<Double> scores = scoringModel.scoreAll(documents, query).content();

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

        String query = "What is a dog?";
        List<TextSegment> documents = List.of(
                TextSegment.from("A dog is a descendant of wolves"),
                TextSegment.from("Hünde sind haustieren"),
                TextSegment.from("狗是宠物"));

        // when
        List<Double> scores = scoringModel.scoreAll(documents, query).content();

        // then
        assertThat(scores)
                .hasSize(1)
                .allMatch(score -> score >= 0);
    }
}
