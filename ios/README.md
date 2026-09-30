# Sims Taste Draft — native iOS 17+

A four-item, tap-to-rank SwiftUI preference draft using the ten categories from `mattyanko/sims_ranker/index.html`. Seven rounds by default, no onboarding, no submit step, and no response-time collection or weighting. No external dependencies, accounts, or network calls.

## Run on an iPhone

1. On a Mac with Xcode 16 or newer, open `TasteDraft.xcodeproj`.
2. Choose the **TasteDraft** scheme and an iOS 17+ simulator, then Run.
3. For a physical iPhone, select your development team in Signing & Capabilities, change the example bundle identifier to your own, connect/select the phone, and Run. Apple signing and a Mac are required; this source project cannot be installed directly from a mobile browser.

Configure `title`, `rounds`, and `items` at the top of `TasteDraft/TasteDraft.swift`. Each item has a stable unique ID, title, SF Symbol, and a persona name. Use at least four items and at least one round. Invalid configuration shows an explanatory screen instead of starting a broken session. With very large item sets, increase the round count: `4 × rounds` is the maximum number of presentations, so seven rounds cannot cover more than 28 items. Sessions are kept in memory; use Share or Export to retain results.

## Interaction

- Four large buttons appear in a 2×2 grid. Tap the best first, then the remaining items in order.
- A numbered badge animates onto each ranked card, with light impact haptics.
- Tap a ranked card again to remove it; later ranks shift up. Undo restores the previous tap/reset; Reset clears the current order.
- After the fourth tap, the app reveals the last badge for 650 ms, then automatically saves the ranking and advances, with a stronger haptic. Undo, reset, or a card edit during that interval cancels the advance.
- Skip discards the whole round without scoring partial taps or inferring rejection. It counts toward the configured round limit.
- The final screen reveals a persona and top three, relative-strength bars, the full observed ranking, native ShareLink, and a CSV file exporter. An all-skipped session produces no fabricated ranking. Play again starts a fresh session.

System colors, native navigation/list controls, scalable fonts, scrolling at large text sizes, explicit VoiceOver rank values, non-color-only badges, and Reduce Motion support are included. Haptics require a supported physical device. The default session is 28 ranking taps and is designed to fit under two minutes; timing is neither recorded nor used as a signal.

## Scoring and selection

A ranking `a > b > c > d` contributes the **full Plackett–Luce likelihood**:

`P = worth(a)/sum(worth(a,b,c,d)) × worth(b)/sum(worth(b,c,d)) × worth(c)/sum(worth(c,d))`

`worth(i) = exp(utility(i))`. The last choice is implicit. The model fits all completed rankings jointly by gradient ascent with backtracking. A zero-mean Gaussian prior (precision 0.5) makes utilities identifiable and finite with sparse, disconnected, or perfectly consistent data. It does not convert each ranking into six independent pairwise votes. The optimizer uses stable log-sum-exp arithmetic.

Uncertainty is approximated by the diagonal of the **inverse full observed information matrix**, including off-diagonal covariances, using Cholesky solves. The selector prioritizes broad exposure early, then combines posterior standard error, exposure balance, and new opponents, with a mild penalty for immediate repeats. Positions are shuffled. This is a practical adaptive heuristic, not a claim of optimal experimental design or statistical convergence after seven rounds.

Results exclude items never present in a completed ranking. A score bar is `100 × exp(utility − leading utility)`; it is relative model worth, **not confidence or a percentage of liking**. Closely matched leaders get an eclectic persona. Seven rounds give a quick, provisional profile; increasing rounds improves evidence. The persona is descriptive game framing, not a diagnostic assessment. CSV contains utilities, relative worth, and ordered item IDs for each completed ranking, with no timing fields.

## Analysis of the original index

The current index implements an exposure-balanced pair generator with ten categories and four target exposures (nominally twenty pairs). It records latency with `performance.now()`, assigns +2/−2 per choice, scales these totals by average latency / individual latency, and min-max normalizes them for display. Despite the requested context, **the checked-in index does not implement a Plackett–Luce optimizer**. Its summary also interprets nonpositive relative totals as dislike, which is stronger than forced-choice data supports.

The pair generator can finish below its target when remaining underexposed items have already been paired. Choice handlers remain active during the scheduled advance and can record multiple choices on rapid taps. The summary writes Markdown markers through `innerHTML`, so bold markers are displayed literally. The native implementation replaces that interaction/state handling and scoring; it does not reuse latency weighting or infer absolute rejection.

## Verification

The app includes an XCTest target for the ranking likelihood, finite-difference gradient/Hessian checks, optimizer convergence, unseen-item behavior, adaptive coverage, undo/reset, skip, duplicate-commit protection, automatic advance/cancellation, and invalid configuration. Run Product → Test (⌘U) in Xcode.

A shared scheme and macOS GitHub Actions workflow are provided. The workflow selects an available iPhone simulator and runs the tests. The Linux implementation environment cannot build SwiftUI/UIKit or execute iOS XCTest. The actual Swift model passed compiler-run checks for its likelihood, gradient/Hessian, optimizer convergence, prior behavior, and coverage across 100 adaptive sessions. The actual session state was also compiled with Observation and a no-op UIKit haptic substitute, then checked for undo/reset, automatic advance and cancellation, skip, restart, duplicate protection, and invalid configuration. Source syntax and project/scheme parsing passed; a simulator build and accessibility/haptics review on an iPhone remain required before release. No App Store icon or distribution signing is included.

Manual review: check light/dark appearance, maximum accessibility text size, VoiceOver rank announcements and focus after each round, Reduce Motion, edits during the completion interval, skips, CSV saving/cancellation, share sheet, and a seven-round session on a physical device.
