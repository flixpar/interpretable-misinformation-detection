using CSV
using DataFrames


println("Reading community notes data")
cn_labels = DataFrame(CSV.File(joinpath(@__DIR__, "../community-notes/processed/community_notes.csv")))
cn_tweets = DataFrame(CSV.File(joinpath(@__DIR__, "../community-notes/processed/community_notes_tweets.csv")))

println("Processing community notes data")
cn_data = innerjoin(cn_tweets, cn_labels, on=:tweetId)
dropmissing!(cn_data, :content)
filter!(r -> r.tweetId == r.conversation_id, cn_data)
rename!(cn_data, :summary => :meta)
select!(cn_data, Not([:conversation_id, :noteId, :score]))
cn_data.source = fill("community-notes", size(cn_data, 1))

println("Reading news tweets data")
news_tweets = DataFrame(CSV.File(joinpath(@__DIR__, "../news-tweets/processed/news_tweets.csv")))

println("Processing news tweets data")
filter!(r -> r.tweetId == r.conversation_id, news_tweets);
select!(news_tweets, Not([:conversation_id]))
rename!(news_tweets, :site => :meta)
news_tweets.misleading = fill(false, nrow(news_tweets))
news_tweets.believable = fill(false, nrow(news_tweets))
news_tweets.harmful = fill(false, nrow(news_tweets))
news_tweets.factual = fill(true, nrow(news_tweets))
news_tweets.satire = fill(false, nrow(news_tweets))
news_tweets.missing_context = fill(false, nrow(news_tweets))
news_tweets.manipulated_media = fill(false, nrow(news_tweets))
news_tweets.opinion = fill(false, nrow(news_tweets))
news_tweets.source = fill("news", nrow(news_tweets));

println("Combining datasets")
tweets = vcat(cn_data, news_tweets)
filter!(r -> length(r.content) > 3, tweets)
sort!(tweets, :tweetId)

mkpath(joinpath(@__DIR__, "processed/"))
CSV.write(joinpath(@__DIR__, "processed/tweets_cn_news.csv"), tweets)
