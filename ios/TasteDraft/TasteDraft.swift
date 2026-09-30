import SwiftUI
import UIKit
import Observation

// MARK: - Reskin here. Stable IDs must be unique; provide at least four items.
enum DraftConfiguration {
    static let title = "Sims Taste Draft"
    static let rounds = 7
    static let items: [DraftItem] = [
        .init(id: "building", title: "Building & creating", symbol: "hammer", persona: "The Architect"),
        .init(id: "characters", title: "Character creation", symbol: "person.crop.square", persona: "The Character Designer"),
        .init(id: "story", title: "Story & roleplay", symbol: "book", persona: "The Storyteller"),
        .init(id: "skills", title: "Skill progression", symbol: "chart.line.uptrend.xyaxis", persona: "The Achiever"),
        .init(id: "family", title: "Family building", symbol: "figure.2.and.child.holdinghands", persona: "The Legacy Builder"),
        .init(id: "chaos", title: "Chaos & experimenting", symbol: "flame", persona: "The Chaos Creator"),
        .init(id: "social", title: "Socializing", symbol: "bubble.left.and.bubble.right", persona: "The Connector"),
        .init(id: "exploring", title: "Exploring & discovery", symbol: "map", persona: "The Explorer"),
        .init(id: "supernatural", title: "Supernatural gameplay", symbol: "sparkles", persona: "The Dreamer"),
        .init(id: "collecting", title: "Collecting items", symbol: "square.grid.2x2", persona: "The Collector")
    ]
}

struct DraftItem: Identifiable, Equatable, Codable {
    let id: String
    let title: String
    let symbol: String
    let persona: String
}

// MARK: - Ranking-only model (no timestamps or response-time weights)
struct RankingObservation: Codable, Equatable {
    let order: [String] // Best to worst, exactly four distinct IDs.
}

struct PreferenceEstimate {
    var utilities: [Double]
    var standardErrors: [Double]
}

/// Gaussian-regularized Plackett–Luce maximum a posteriori estimate.
/// log P(order) = sum over the first three choices of u[winner] - logsumexp(u[remaining]).
/// The proper prior keeps estimates finite for undefeated items and disconnected data.
enum PlackettLuce {
    static let priorPrecision = 0.5

    static func fit(items: [DraftItem], observations: [RankingObservation]) -> PreferenceEstimate {
        let count = items.count
        guard count > 0 else { return .init(utilities: [], standardErrors: []) }
        let indices = Dictionary(uniqueKeysWithValues: items.enumerated().map { ($0.element.id, $0.offset) })
        let rankings = observations.compactMap { observation -> [Int]? in
            let order = observation.order.compactMap { indices[$0] }
            return order.count == 4 && Set(order).count == 4 ? order : nil
        }
        var utilities = Array(repeating: 0.0, count: count)
        // Gradient ascent with backtracking, using the actual joint ranking likelihood.
        for _ in 0..<250 {
            let current = derivatives(utilities, rankings)
            if (current.gradient.map({ abs($0) }).max() ?? 0) < 1e-7 { break }
            var step = 1.0
            var accepted = false
            for _ in 0..<24 {
                let candidate = zip(utilities, current.gradient).map { $0 + step * $1 }
                let next = derivatives(candidate, rankings).value
                let norm = current.gradient.reduce(0) { $0 + $1 * $1 }
                if next >= current.value + 0.0001 * step * norm {
                    utilities = candidate
                    accepted = true
                    break
                }
                step *= 0.5
            }
            if !accepted { break }
        }
        let information = derivatives(utilities, rankings).information
        // Full inverse information, including cross-item covariance, via Cholesky solves.
        let covariance = inversePositiveDefinite(information)
        return .init(utilities: utilities,
                     standardErrors: (0..<count).map { sqrt(max(0, covariance[$0][$0])) })
    }

    static func derivatives(_ utilities: [Double], _ rankings: [[Int]])
        -> (value: Double, gradient: [Double], information: [[Double]]) {
        let count = utilities.count
        var value = -0.5 * priorPrecision * utilities.reduce(0) { $0 + $1 * $1 }
        var gradient = utilities.map { -priorPrecision * $0 }
        var information = Array(repeating: Array(repeating: 0.0, count: count), count: count)
        for i in 0..<count { information[i][i] = priorPrecision }
        for ranking in rankings {
            for stage in 0..<(ranking.count - 1) {
                let remaining = Array(ranking[stage...])
                let maximum = remaining.map { utilities[$0] }.max()!
                let weights = remaining.map { exp(utilities[$0] - maximum) }
                let total = weights.reduce(0, +)
                let probabilities = weights.map { $0 / total }
                value += utilities[ranking[stage]] - maximum - log(total)
                gradient[ranking[stage]] += 1
                for (a, i) in remaining.enumerated() {
                    gradient[i] -= probabilities[a]
                    for (b, j) in remaining.enumerated() {
                        information[i][j] += (i == j ? probabilities[a] : 0) - probabilities[a] * probabilities[b]
                    }
                }
            }
        }
        return (value, gradient, information)
    }

    private static func inversePositiveDefinite(_ matrix: [[Double]]) -> [[Double]] {
        let count = matrix.count
        var lower = Array(repeating: Array(repeating: 0.0, count: count), count: count)
        for i in 0..<count {
            for j in 0...i {
                var value = matrix[i][j]
                for k in 0..<j { value -= lower[i][k] * lower[j][k] }
                lower[i][j] = i == j ? sqrt(max(value, 1e-12)) : value / lower[j][j]
            }
        }
        var inverse = lower
        for column in 0..<count {
            var forward = Array(repeating: 0.0, count: count)
            for i in 0..<count {
                var value = i == column ? 1.0 : 0.0
                for k in 0..<i { value -= lower[i][k] * forward[k] }
                forward[i] = value / lower[i][i]
            }
            var solution = forward
            for i in stride(from: count - 1, through: 0, by: -1) {
                var value = forward[i]
                if i + 1 < count {
                    for k in (i + 1)..<count { value -= lower[k][i] * solution[k] }
                }
                solution[i] = value / lower[i][i]
            }
            for i in 0..<count { inverse[i][column] = solution[i] }
        }
        return inverse
    }
}

/// Coverage first; then uncertainty, exposure balance, and opponent diversity.
/// Random jitter and shuffled positions avoid deterministic presentation bias.
enum AdaptiveSelector {
    static func select(items: [DraftItem], observations: [RankingObservation],
                       presentations: [[String]], estimate: PreferenceEstimate,
                       previous: [String], random: () -> Double = { Double.random(in: 0..<1) }) -> [String] {
        guard items.count >= 4 else { return [] }
        let appearances = items.map { item in presentations.filter { $0.contains(item.id) }.count }
        let evidence = items.map { item in observations.filter { $0.order.contains(item.id) }.count }
        let broad = observations.count < max(2, Int(ceil(Double(items.count) / 4)))
        let jitter = items.map { _ in 0.25 * random() }
        var chosen: [Int] = []
        while chosen.count < 4 {
            let candidates = items.indices.filter { !chosen.contains($0) }
            let candidate = candidates.max { a, b in
                priority(a) < priority(b)
            }!
            chosen.append(candidate)
        }
        return chosen.map { items[$0].id }.shuffled()

        func priority(_ index: Int) -> Double {
            let item = items[index]
            let novelOpponents = chosen.reduce(0.0) { total, other in
                let coappearances = observations.filter {
                    $0.order.contains(item.id) && $0.order.contains(items[other].id)
                }.count
                return total + 1.0 / Double(1 + coappearances)
            }
            let uncertainty = estimate.standardErrors[index] / sqrt(1 / PlackettLuce.priorPrecision)
            // Coverage dominates early. Later, uncertainty dominates but coverage remains protected.
            let coverage = broad ? 6.0 : 2.0
            let unseen = evidence[index] == 0 ? 4.0 : 0.0
            let repeatPenalty = previous.contains(item.id) && items.count > 4 ? 0.8 : 0
            // A tiny fixed jitter should be generated once per candidate, not during comparisons.
            return coverage / Double(1 + appearances[index]) + unseen + (broad ? 0 : 3 * uncertainty)
                + 1.2 * novelOpponents - repeatPenalty + jitter[index]
        }
    }
}

// MARK: - Session state
@MainActor @Observable
final class DraftSession {
    let items: [DraftItem]
    let roundLimit: Int
    var cards: [String] = []
    var order: [String] = []
    var observations: [RankingObservation] = []
    var presentations: [[String]] = []
    var skippedRounds = 0
    var finished = false
    var completing = false
    var generation = 0
    var estimate: PreferenceEstimate
    private var history: [[String]] = []
    private var completionTask: Task<Void, Never>?

    init(items: [DraftItem] = DraftConfiguration.items, rounds: Int = DraftConfiguration.rounds) {
        self.items = items
        roundLimit = rounds
        estimate = .init(utilities: Array(repeating: 0, count: items.count),
                         standardErrors: Array(repeating: sqrt(1 / PlackettLuce.priorPrecision), count: items.count))
        if configurationError == nil { nextRound() }
    }

    var configurationError: String? {
        if items.count < 4 { return "Add at least four items in DraftConfiguration." }
        if Set(items.map(\.id)).count != items.count { return "Each configured item needs a unique ID." }
        if items.contains(where: { $0.id.isEmpty || $0.title.isEmpty }) { return "Items need nonempty IDs and titles." }
        if roundLimit < 1 { return "Set the round count to at least one." }
        return nil
    }
    var completedRounds: Int { observations.count + skippedRounds }
    var roundNumber: Int { min(completedRounds + 1, roundLimit) }
    var canUndo: Bool { !history.isEmpty }
    var rankedItems: [DraftItem] {
        items.enumerated().filter { entry in observations.contains { $0.order.contains(entry.element.id) } }
            .sorted { a, b in
                let difference = estimate.utilities[a.offset] - estimate.utilities[b.offset]
                return abs(difference) < 1e-8 ? a.offset < b.offset : difference > 0
            }.map(\.element)
    }
    var hasClearLeader: Bool {
        let ranked = rankedItems
        guard ranked.count >= 2 else { return false }
        return abs(utility(ranked[0]) - utility(ranked[1])) > 0.05
    }
    var persona: String {
        guard !rankedItems.isEmpty else { return "A taste still to discover" }
        return hasClearLeader ? rankedItems[0].persona : "The Eclectic Player"
    }
    func utility(_ item: DraftItem) -> Double { estimate.utilities[items.firstIndex(of: item)!] }
    func bar(_ item: DraftItem) -> Double {
        guard let top = rankedItems.first else { return 0 }
        // Worth relative to the leading item, not a percentage of liking or confidence.
        return exp(utility(item) - utility(top))
    }

    func tap(_ id: String) {
        guard !finished, cards.contains(id) else { return }
        cancelCompletion()
        history.append(order)
        if let index = order.firstIndex(of: id) { order.remove(at: index) }
        else { order.append(id) }
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
        if order.count == 4 { scheduleCompletion() }
    }
    func undo() {
        guard let previous = history.popLast() else { return }
        cancelCompletion()
        order = previous
        if order.count == 4 { scheduleCompletion() }
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }
    func resetRound() {
        guard !order.isEmpty else { return }
        cancelCompletion()
        history.append(order)
        order = []
        UIImpactFeedbackGenerator(style: .light).impactOccurred()
    }
    func skip() {
        guard !finished else { return }
        cancelCompletion()
        skippedRounds += 1
        advance()
    }
    func restart() {
        cancelCompletion()
        observations = []
        presentations = []
        skippedRounds = 0
        finished = false
        estimate = PlackettLuce.fit(items: items, observations: [])
        nextRound()
    }
    func cancelCompletion() {
        completionTask?.cancel()
        completionTask = nil
        completing = false
    }
    private func scheduleCompletion() {
        completing = true
        // Brief badge reveal; no submission or confirmation. Edits cancel the pending advance.
        completionTask = Task { [weak self] in
            do { try await Task.sleep(for: .milliseconds(650)) } catch { return }
            guard let self, !Task.isCancelled, self.order.count == 4, !self.finished else { return }
            self.commitRound()
        }
    }
    func commitRound() {
        guard order.count == 4, Set(order) == Set(cards), !finished else { return }
        completionTask?.cancel()
        completionTask = nil
        observations.append(.init(order: order))
        estimate = PlackettLuce.fit(items: items, observations: observations)
        UIImpactFeedbackGenerator(style: .heavy).impactOccurred()
        advance()
    }
    private func advance() {
        completing = false
        order = []
        history = []
        if completedRounds >= roundLimit { finished = true }
        else { nextRound() }
    }
    private func nextRound() {
        let previous = cards
        cards = AdaptiveSelector.select(items: items, observations: observations,
                                        presentations: presentations, estimate: estimate, previous: previous)
        presentations.append(cards)
        order = []
        history = []
        generation += 1
    }
    var summary: String {
        let top = rankedItems.prefix(3).enumerated().map { "\($0.offset + 1). \($0.element.title)" }.joined(separator: "\n")
        return "\(DraftConfiguration.title)\n\(persona)\n\n\(top)\n\n\(observations.count) ranked rounds; \(skippedRounds) skipped.\nBased on tap order alone. A relative preference snapshot."
    }
    var csv: String {
        func quoted(_ value: String) -> String { "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }
        var lines = ["rank,item_id,item,utility,relative_worth"]
        for (index, item) in rankedItems.enumerated() {
            lines.append("\(index + 1),\(quoted(item.id)),\(quoted(item.title)),\(utility(item)),\(bar(item))")
        }
        lines += ["", "observation,rank_1_id,rank_2_id,rank_3_id,rank_4_id"]
        for (index, observation) in observations.enumerated() {
            lines.append(([String(index + 1)] + observation.order.map(quoted)).joined(separator: ","))
        }
        lines += ["", "ranked_rounds,skipped_rounds", "\(observations.count),\(skippedRounds)"]
        return lines.joined(separator: "\n")
    }
}

// MARK: - Native iPhone interface
@main
struct TasteDraftApp: App {
    var body: some Scene { WindowGroup { DraftView() } }
}

struct DraftView: View {
    @State private var session = DraftSession()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @AccessibilityFocusState private var instructionFocused: Bool

    var body: some View {
        NavigationStack {
            Group {
                if let error = session.configurationError {
                    ContentUnavailableView("Check your item set", systemImage: "exclamationmark.triangle", description: Text(error))
                } else if session.finished {
                    ResultsView(session: session)
                        .transition(.opacity)
                } else {
                    roundView
                        .transition(.opacity)
                }
            }
            .navigationTitle(DraftConfiguration.title)
            .navigationBarTitleDisplayMode(.inline)
            .background(Color(uiColor: .systemGroupedBackground))
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.25), value: session.finished)
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: session.generation)
        }
    }
    private var roundView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Round \(session.roundNumber) of \(session.roundLimit)")
                        .font(.subheadline.weight(.semibold)).foregroundStyle(.secondary)
                    ProgressView(value: Double(session.completedRounds), total: Double(session.roundLimit))
                        .accessibilityLabel("Draft progress")
                        .accessibilityValue("\(session.completedRounds) of \(session.roundLimit) rounds complete")
                    Text("Tap your favorites in order")
                        .font(.title2.bold())
                        .accessibilityFocused($instructionFocused)
                    Text(session.completing ? "Round complete" : "\(session.order.count) of 4 ranked")
                        .font(.subheadline).foregroundStyle(.secondary)
                }
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                    ForEach(session.cards, id: \.self) { id in
                        if let item = session.items.first(where: { $0.id == id }) {
                            RankCard(item: item, rank: session.order.firstIndex(of: id).map { $0 + 1 }) {
                                withAnimation(reduceMotion ? nil : .spring(response: 0.28, dampingFraction: 0.65)) { session.tap(id) }
                            }
                        }
                    }
                }.id(session.generation)
                HStack(spacing: 24) {
                    Button { withAnimation(reduceMotion ? nil : .default) { session.undo() } } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                        .disabled(!session.canUndo)
                    Spacer(minLength: 0)
                    Button("Reset") { withAnimation(reduceMotion ? nil : .default) { session.resetRound() } }
                        .disabled(session.order.isEmpty)
                }.buttonStyle(.bordered).controlSize(.large)
                Button { session.skip() } label: {
                    Label("Skip round — unfamiliar items", systemImage: "forward.end")
                        .frame(maxWidth: .infinity).padding(.vertical, 8)
                }
                .accessibilityHint("Skips these four items without recording a preference.")
                .padding(.bottom, 16)
            }.padding()
        }
        .onChange(of: session.generation) { _, _ in instructionFocused = true }
    }
}

struct RankCard: View {
    let item: DraftItem
    let rank: Int?
    let action: () -> Void
    @ScaledMetric(relativeTo: .body) private var minimumHeight = 166.0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Image(systemName: item.symbol).font(.title2).foregroundStyle(.primary)
                        .accessibilityHidden(true)
                    Spacer(minLength: 4)
                    ZStack {
                        if let rank {
                            Text("\(rank)").font(.headline.bold()).monospacedDigit()
                                .foregroundStyle(Color(uiColor: .systemBackground))
                                .frame(width: 34, height: 34)
                                .background(Color.primary, in: Circle())
                                .transition(reduceMotion ? .opacity : .scale.combined(with: .opacity))
                                .id(rank)
                        }
                    }.frame(width: 34, height: 34)
                }
                Text(item.title).font(.headline).foregroundStyle(.primary)
                    .multilineTextAlignment(.leading).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .padding(16).frame(maxWidth: .infinity, alignment: .leading)
            .frame(minHeight: minimumHeight, alignment: .topLeading)
            .background(Color(uiColor: .secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 16))
            .overlay {
                RoundedRectangle(cornerRadius: 16).strokeBorder(rank == nil ? Color.clear : Color.primary, lineWidth: 2)
            }
            .contentShape(RoundedRectangle(cornerRadius: 16))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(item.title)
        .accessibilityValue(rank.map { "Rank \($0) of 4" } ?? "Not ranked")
        .accessibilityHint(rank == nil ? "Double tap to assign the next preference rank." : "Double tap to remove this rank. Later ranks move up.")
    }
}

struct ResultsView: View {
    let session: DraftSession
    @State private var revealed = false
    @State private var exporting = false
    @State private var exportError: String?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 12) {
                    Image(systemName: session.rankedItems.isEmpty ? "questionmark.circle" : "trophy.fill")
                        .font(.largeTitle).foregroundStyle(.primary).accessibilityHidden(true)
                    Text(session.rankedItems.isEmpty ? "No rankings yet" : "Your taste profile").font(.subheadline).foregroundStyle(.secondary)
                    Text(session.persona).font(.largeTitle.bold()).fixedSize(horizontal: false, vertical: true)
                    Text(profileDescription).foregroundStyle(.secondary)
                }.padding(.vertical, 12)
                    .opacity(revealed ? 1 : 0)
                    .offset(y: revealed || reduceMotion ? 0 : 16)
            }
            if !session.rankedItems.isEmpty {
                Section("Your top picks") {
                    ForEach(Array(session.rankedItems.prefix(3).enumerated()), id: \.element.id) { index, item in
                        scoreRow(item, rank: index + 1)
                    }
                }
                Section {
                    DisclosureGroup("Full ranking") {
                        ForEach(Array(session.rankedItems.enumerated()), id: \.element.id) { index, item in
                            scoreRow(item, rank: index + 1)
                        }
                    }
                } footer: {
                    Text("Bars show relative preference strength, with the leading item set to 100. They are not confidence or liking percentages. Only ranked items are included; close scores can change with more rounds.")
                }
            }
            Section {
                ShareLink(item: session.summary) { Label("Share taste profile", systemImage: "square.and.arrow.up") }
                Button { exporting = true } label: { Label("Export rankings (CSV)", systemImage: "square.and.arrow.down") }
                Button { session.restart() } label: { Label("Play again", systemImage: "arrow.clockwise") }
            }
        }
        .listStyle(.insetGrouped)
        .fileExporter(isPresented: $exporting, document: CSVDocument(text: session.csv), contentType: .commaSeparatedText,
                      defaultFilename: "taste-draft") { result in
            if case .failure(let error) = result { exportError = error.localizedDescription }
        }
        .alert("Couldn’t export rankings", isPresented: Binding(
            get: { exportError != nil }, set: { if !$0 { exportError = nil } }
        )) {
            Button("OK", role: .cancel) { exportError = nil }
        } message: { Text(exportError ?? "Please try again.") }
        .task {
            withAnimation(reduceMotion ? nil : .easeOut(duration: 0.6)) { revealed = true }
        }
    }
    private var profileDescription: String {
        guard let top = session.rankedItems.first else {
            return "All rounds were skipped. Play again to discover your favorites."
        }
        let coverage = "\(session.observations.count) rounds ranked · \(session.skippedRounds) skipped · \(session.rankedItems.count) of \(session.items.count) items compared."
        return (session.hasClearLeader ? "Your draft puts \(top.title.lowercased()) first. " : "Your leading picks are closely matched. ") + coverage
    }
    private func scoreRow(_ item: DraftItem, rank: Int) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text("\(rank).").font(.headline).monospacedDigit()
                Text(item.title).font(.body.weight(.semibold))
                Spacer(minLength: 8)
                Text("\(Int((session.bar(item) * 100).rounded()))").font(.subheadline).monospacedDigit().foregroundStyle(.secondary)
            }
            ProgressView(value: revealed ? session.bar(item) : 0).tint(.primary)
                .accessibilityHidden(true)
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(rank). \(item.title)")
        .accessibilityValue("Relative preference strength \(Int((session.bar(item) * 100).rounded())) out of 100")
    }
}

import UniformTypeIdentifiers
struct CSVDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.commaSeparatedText] }
    var text: String
    init(text: String) { self.text = text }
    init(configuration: ReadConfiguration) throws {
        text = String(decoding: configuration.file.regularFileContents ?? Data(), as: UTF8.self)
    }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

#Preview { DraftView() }
