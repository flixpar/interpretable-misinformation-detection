using Downloads
using Dates

using CSV
using DataFrames
using Statistics


SCORE_THRESHOLD = 0.4
MIN_REVIEWS = 4

function process_community_notes()
	notes_data = DataFrame(CSV.File("downloads/community_notes.tsv", delim='\t'))

	notes_data.misleading = notes_data.classification .== "MISINFORMED_OR_POTENTIALLY_MISLEADING"
	notes_data.believable = notes_data.believable .== "BELIEVABLE_BY_MANY"
	notes_data.harmful = notes_data.harmful .== "CONSIDERABLE_HARM"
	notes_data.factual = (1 .- notes_data.misleadingFactualError) + notes_data.notMisleadingFactuallyCorrect
	notes_data.satire = notes_data.misleadingSatire + notes_data.notMisleadingClearlySatire

	select!(
		notes_data,
		:tweetId,
		:noteId,
		:misleading,
		:believable,
		:harmful,
		:factual,
		:satire,
		:misleadingMissingImportantContext => :missing_context,
		:misleadingManipulatedMedia => :manipulated_media,
		:notMisleadingPersonalOpinion => :opinion,
		:summary,
	)

	scores = compute_scores()
	insertcols!(notes_data, 3, :score => map(r -> get(scores, r, 0), notes_data.noteId))
	sort!(notes_data, [:noteId, :score], rev=[false, true])
	unique!(notes_data, :tweetId)

	filter!(r -> r.score > SCORE_THRESHOLD, notes_data)
	println("Labeled $(nrow(notes_data)) tweets")

	notes_data |> CSV.write("processed/community_notes.csv")

	return notes_data
end

function compute_scores()
	ratings_data = DataFrame(CSV.File("downloads/ratings.tsv", delim='\t'))

	ratings_data.helpfulnessValue = map(r -> begin
		if !ismissing(r.helpfulnessLevel)
			if r.helpfulnessLevel == "HELPFUL"
				return 1
			elseif r.helpfulnessLevel == "SOMEWHAT_HELPFUL"
				return 0.5
			else
				return 0
			end
		else
			return r.helpful
		end
	end, eachrow(ratings_data))

	avg_ratings = combine(
		groupby(ratings_data, :noteId),
		:agree => mean => :avg_agreement,
		:helpfulnessValue => mean => :avg_helpfulness,
		:agree => length => :n_ratings,
	)

	function score_metric(r)
		if r.n_ratings < MIN_REVIEWS
			return 0
		end
		score = 0.5 * r.avg_agreement + 0.5 * r.avg_helpfulness
		return score
	end
	avg_ratings.score = map(score_metric, eachrow(avg_ratings))

	# ratings = Dict(
	# 	r.noteId => (;
	# 		r.score,
	# 		r.avg_agreement,
	# 		r.avg_helpfulness,
	# 		r.n_ratings,
	# 	) for r in eachrow(avg_ratings)
	# )
	# return ratings

	scores = Dict(r.noteId => r.score for r in eachrow(avg_ratings))
	return scores
end

function download_community_notes()
	date = today()
	date_str = Dates.format(date, "yyyy/mm/dd")
	Downloads.download("https://ton.twimg.com/birdwatch-public-data/$(date_str)/notes/notes-00000.tsv", "downloads/community_notes.tsv")
	Downloads.download("https://ton.twimg.com/birdwatch-public-data/$(date_str)/noteRatings/ratings-00000.tsv", "downloads/ratings.tsv")
	Downloads.download("https://ton.twimg.com/birdwatch-public-data/$(date_str)/noteStatusHistory/noteStatusHistory-00000.tsv", "downloads/status_history.tsv")
	Downloads.download("https://ton.twimg.com/birdwatch-public-data/$(date_str)/userEnrollment/userEnrollment-00000.tsv", "downloads/user_enrollment.tsv")
	return
end

if abspath(PROGRAM_FILE) == @__FILE__
	download_community_notes()
	process_community_notes()
end
