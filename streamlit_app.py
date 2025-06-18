import streamlit as st
from app import SimpleURLAnalyzer
import json

st.set_page_config(page_title="Review Analyzer", layout="wide")

st.title("Review Analyzer Web App")
st.write("Analyze reviews from Trustpilot, IMDb, Steam, or Google Play Store URLs.")

# User input
url = st.text_input("Enter a review page URL:", "")
max_reviews = st.number_input("Max reviews to analyze", min_value=1, max_value=100, value=20)

if st.button("Analyze"):
    if not url:
        st.warning("Please enter a URL.")
    else:
        with st.spinner("Analyzing..."):
            analyzer = SimpleURLAnalyzer()
            results = analyzer.analyze_url(url, max_reviews)
        
        if "error" in results:
            st.error(f"Error: {results['error']}")
        else:
            summary = results.get("summary", {})
            st.subheader("Summary")

            # Show key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Platform", summary.get("platform", "N/A"))
            col2.metric("Content Name", summary.get("content_name", "N/A"))
            col3.metric("Total Reviews", summary.get("total_reviews_analyzed", "N/A"))
            col4.metric("Avg. Confidence", f"{summary.get('average_confidence', 0):.2f}")

            col5, col6, col7 = st.columns(3)
            col5.metric("Avg. Sentiment Score", f"{summary.get('average_sentiment_score', 0):.2f}")
            col6.metric("Avg. Predicted Rating", f"{summary.get('average_predicted_rating', 0):.2f}")
            col7.metric("Most Common Emotion", summary.get("most_common_emotion", "N/A"))

            # Sentiment distribution
            st.markdown("**Sentiment Distribution:**")
            sent_dist = summary.get("sentiment_distribution", {})
            st.write(sent_dist)

            # Top keywords
            st.markdown("**Top Keywords:**")
            st.write(", ".join(summary.get("top_keywords", [])))

            # Top liked/disliked points
            st.markdown("**Top 3 Liked Points:**")
            for point in summary.get("top_3_liked", []):
                st.success(point)

            st.markdown("**Top 3 Disliked Points:**")
            for point in summary.get("top_3_disliked", []):
                st.error(point)
            
            # Show reviews in a table
            reviews = results.get("reviews", [])
            if reviews:
                st.subheader("Individual Reviews")
                st.dataframe([
                    {
                        "Review #": r.get("review_id"),
                        "Text": r.get("text"),
                        "Sentiment": r.get("sentiment", {}).get("label"),
                        "Predicted Rating": r.get("predicted_rating"),
                        "Top Emotion": r.get("emotions", {}).get("top_emotion"),
                        "Keywords": ", ".join(r.get("keywords", [])),
                        "Date": r.get("date", "")
                    }
                    for r in reviews
                ])
            
            # Optionally, allow download of results
            st.download_button(
                label="Download Results as JSON",
                data=json.dumps(results, indent=2, ensure_ascii=False),
                file_name="analysis_results.json",
                mime="application/json"
            )
