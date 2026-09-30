import XCTest
@testable import TasteDraft

final class TasteDraftTests: XCTestCase {
    let items = DraftConfiguration.items

    func testFullRankingLikelihoodAtEqualWorth() {
        let result = PlackettLuce.derivatives([0, 0, 0, 0], [[0, 1, 2, 3]])
        XCTAssertEqual(result.value, -log(24), accuracy: 1e-10)
    }

    func testAnalyticalGradientMatchesFiniteDifference() {
        let utilities = [0.7, -0.3, 0.2, -0.6]
        let rankings = [[2, 0, 3, 1], [0, 1, 2, 3]]
        let derivative = PlackettLuce.derivatives(utilities, rankings)
        for i in utilities.indices {
            var plus = utilities, minus = utilities
            plus[i] += 1e-5
            minus[i] -= 1e-5
            let numerical = (PlackettLuce.derivatives(plus, rankings).value - PlackettLuce.derivatives(minus, rankings).value) / 2e-5
            XCTAssertEqual(derivative.gradient[i], numerical, accuracy: 1e-6)
            let gradientPlus = PlackettLuce.derivatives(plus, rankings).gradient
            let gradientMinus = PlackettLuce.derivatives(minus, rankings).gradient
            for j in utilities.indices {
                XCTAssertEqual(derivative.information[j][i], -(gradientPlus[j] - gradientMinus[j]) / 2e-5, accuracy: 1e-6)
            }
        }
    }

    func testRepeatedRankingConvergesWithFiniteUtilities() {
        let subset = Array(items.prefix(4))
        let observation = RankingObservation(order: subset.map(\.id))
        let result = PlackettLuce.fit(items: subset, observations: Array(repeating: observation, count: 30))
        for i in 0..<3 { XCTAssertGreaterThan(result.utilities[i], result.utilities[i + 1]) }
        XCTAssertTrue(result.utilities.allSatisfy(\.isFinite))
        XCTAssertTrue(result.standardErrors.allSatisfy { $0.isFinite && $0 > 0 })
        let residual = PlackettLuce.derivatives(result.utilities, Array(repeating: [0, 1, 2, 3], count: 30))
        XCTAssertLessThan(residual.gradient.map { abs($0) }.max()!, 1e-5)
    }

    func testUnseenItemsStayAtPriorAndBalancedRankingsTie() {
        let forward = RankingObservation(order: Array(items.prefix(4)).map(\.id))
        let reverse = RankingObservation(order: Array(items.prefix(4)).reversed().map(\.id))
        let result = PlackettLuce.fit(items: items, observations: [forward, reverse])
        for i in 4..<items.count {
            XCTAssertEqual(result.utilities[i], 0, accuracy: 1e-8)
            XCTAssertEqual(result.standardErrors[i], sqrt(2), accuracy: 1e-8)
        }
        XCTAssertEqual(result.utilities[0], result.utilities[3], accuracy: 1e-6)
        XCTAssertEqual(result.utilities[1], result.utilities[2], accuracy: 1e-6)
    }

    func testSelectorCoversAllItemsAndUsesFourUniqueCards() {
        var observations: [RankingObservation] = []
        var presentations: [[String]] = []
        var previous: [String] = []
        for _ in 0..<7 {
            let estimate = PlackettLuce.fit(items: items, observations: observations)
            let cards = AdaptiveSelector.select(items: items, observations: observations,
                                                presentations: presentations, estimate: estimate,
                                                previous: previous, random: { 0.5 })
            XCTAssertEqual(cards.count, 4)
            XCTAssertEqual(Set(cards).count, 4)
            observations.append(.init(order: cards))
            presentations.append(cards)
            previous = cards
        }
        XCTAssertEqual(Set(presentations.flatMap { $0 }), Set(items.map(\.id)))
    }

    @MainActor func testRemoveCompactsRanksAndUndoRestoresState() {
        let session = DraftSession(items: Array(items.prefix(4)), rounds: 2)
        let cards = session.cards
        session.tap(cards[0]); session.tap(cards[1]); session.tap(cards[2])
        session.tap(cards[1])
        XCTAssertEqual(session.order, [cards[0], cards[2]])
        session.undo()
        XCTAssertEqual(session.order, Array(cards.prefix(3)))
        session.resetRound()
        XCTAssertTrue(session.order.isEmpty)
        session.undo()
        XCTAssertEqual(session.order, Array(cards.prefix(3)))
        session.cancelCompletion()
    }

    @MainActor func testSkipDoesNotCreateObservationAndAllSkippedHasNoPersona() {
        let session = DraftSession(items: items, rounds: 2)
        session.tap(session.cards[0])
        session.skip(); session.skip()
        XCTAssertTrue(session.finished)
        XCTAssertTrue(session.observations.isEmpty)
        XCTAssertTrue(session.rankedItems.isEmpty)
        XCTAssertEqual(session.skippedRounds, 2)
    }

    @MainActor func testCompletionCommitsOnceAndRestartClearsEvidence() {
        let session = DraftSession(items: Array(items.prefix(4)), rounds: 1)
        for id in session.cards { session.tap(id) }
        session.commitRound(); session.commitRound()
        XCTAssertEqual(session.observations.count, 1)
        XCTAssertTrue(session.finished)
        XCTAssertFalse(session.csv.contains("latency"))
        session.restart()
        XCTAssertTrue(session.observations.isEmpty)
        XCTAssertFalse(session.finished)
        XCTAssertEqual(session.cards.count, 4)
    }

    @MainActor func testUndoCancelsAutomaticAdvance() async throws {
        let session = DraftSession(items: Array(items.prefix(4)), rounds: 1)
        for id in session.cards { session.tap(id) }
        session.undo()
        try await Task.sleep(for: .milliseconds(800))
        XCTAssertFalse(session.finished)
        XCTAssertTrue(session.observations.isEmpty)
        XCTAssertEqual(session.order.count, 3)
    }

    @MainActor func testAutomaticAdvance() async throws {
        let session = DraftSession(items: Array(items.prefix(4)), rounds: 1)
        for id in session.cards { session.tap(id) }
        try await Task.sleep(for: .milliseconds(900))
        XCTAssertTrue(session.finished)
        XCTAssertEqual(session.observations.count, 1)
    }

    @MainActor func testInvalidConfigurationDoesNotCrash() {
        let session = DraftSession(items: Array(repeating: items[0], count: 4), rounds: 7)
        XCTAssertNotNil(session.configurationError)
        XCTAssertTrue(session.cards.isEmpty)
    }
}
