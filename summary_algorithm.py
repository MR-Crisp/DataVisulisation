import numpy as np
import pandas as pd


class summaryAlgorithm:
    def __init__(self):
        # Initialise the summary algorithm
        pass

    def cluster_sizes(self, labels):
        # Count how many data points belong to each cluster
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

    def cluster_centres_summary(self, gmm):
        # Extract the centroid of each cluster from the GMM
        summaries = []
        for i, centre in enumerate(gmm.means_):
            summaries.append({
                "cluster": i,
                "centre": centre.tolist()
            })
        return summaries

    def average_distance_to_centroid(self, X, labels, gmm):
        # Measure how compact each cluster is
        # Lower average distance means the cluster is tighter
        results = {}

        for cluster_id in np.unique(labels):
            cluster_points = X[labels == cluster_id]
            centroid = gmm.means_[cluster_id]

            # Compute Euclidean distance between each point and centroid
            distances = np.linalg.norm(cluster_points - centroid, axis=1)

            # Store the average distance for this cluster
            results[int(cluster_id)] = float(np.mean(distances))

        return results

    def feature_importance(self, X, gmm):
        # Compare each cluster centroid to the global dataset mean
        # This helps identify which latent dimensions make a cluster stand out
        global_mean = np.mean(X, axis=0)
        feature_summaries = {}

        for i, centre in enumerate(gmm.means_):
            diff = centre - global_mean
            feature_summaries[i] = diff

        return feature_summaries

    def describe_features(self, diff, top_k=2, threshold=0.3):
        # Turn feature differences into readable text descriptions
        # Example: "higher values in latent dimension 2"
        diff = np.array(diff)
        abs_diff = np.abs(diff)

        # Rank dimensions by strongest difference first
        ranked_idx = np.argsort(abs_diff)[::-1]

        higher_dims = []
        lower_dims = []

        for idx in ranked_idx:
            # Ignore very small differences
            if abs_diff[idx] < threshold:
                continue

            # Positive means above dataset mean, negative means below
            if diff[idx] > 0:
                higher_dims.append(f"{idx + 1}")
            else:
                lower_dims.append(f"{idx + 1}")

            # Limit how many standout dimensions are described
            if len(higher_dims) + len(lower_dims) == top_k:
                break

        parts = []

        if higher_dims:
            if len(higher_dims) == 1:
                parts.append(f"higher values in latent dimension {higher_dims[0]}")
            else:
                parts.append(
                    f"higher values in latent dimensions {', '.join(higher_dims)}"
                )

        if lower_dims:
            if len(lower_dims) == 1:
                parts.append(f"lower values in latent dimension {lower_dims[0]}")
            else:
                parts.append(
                    f"lower values in latent dimensions {', '.join(lower_dims)}"
                )

        if not parts:
            return "no strongly distinguishing latent dimensions"

        return " and ".join(parts)

    def nearest_cluster(self, gmm):
        # Find the closest cluster to each cluster using centroid distance
        centres = gmm.means_
        nearest = {}

        for i in range(len(centres)):
            distances = []

            for j in range(len(centres)):
                if i == j:
                    continue

                # Euclidean distance between two cluster centres
                dist = np.linalg.norm(centres[i] - centres[j])
                distances.append((j, dist))

            # Select the nearest cluster
            closest_cluster, closest_dist = min(distances, key=lambda x: x[1])

            nearest[i] = {
                "nearest_cluster": int(closest_cluster),
                "distance": float(closest_dist)
            }

        return nearest

    def cluster_similarity_analysis(self, gmm):
        # Build similarity and distance matrices between cluster centres
        centres = gmm.means_
        n_clusters = len(centres)

        similarity_matrix = np.zeros((n_clusters, n_clusters))
        distance_matrix = np.zeros((n_clusters, n_clusters))

        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                    distance_matrix[i, j] = 0.0
                else:
                    # Euclidean distance between centroids
                    distance = np.linalg.norm(centres[i] - centres[j])
                    distance_matrix[i, j] = distance

                    # Convert distance into a similarity score
                    # Smaller distance gives higher similarity
                    similarity_matrix[i, j] = 1 / (1 + distance)

        return similarity_matrix, distance_matrix

    def similarity_summary(self, gmm):
        # Summarise which clusters are most and least similar
        similarity_matrix, distance_matrix = self.cluster_similarity_analysis(gmm)
        n_clusters = similarity_matrix.shape[0]

        most_similar_pair = None
        least_similar_pair = None
        max_similarity = -1
        min_similarity = float("inf")

        nearest_neighbour_summary = []

        for i in range(n_clusters):
            best_j = None
            best_similarity = -1

            for j in range(n_clusters):
                if i == j:
                    continue

                sim = similarity_matrix[i, j]

                # Track the most similar cluster for cluster i
                if sim > best_similarity:
                    best_similarity = sim
                    best_j = j

                # Track the global most similar pair
                if sim > max_similarity:
                    max_similarity = sim
                    most_similar_pair = (i, j)

                # Track the global least similar pair
                if sim < min_similarity:
                    min_similarity = sim
                    least_similar_pair = (i, j)

            nearest_neighbour_summary.append({
                "cluster": i,
                "most_similar_cluster": int(best_j),
                "similarity_score": float(best_similarity),
                "distance": float(distance_matrix[i, best_j])
            })

        return {
            "most_similar_pair": most_similar_pair,
            "most_similar_score": float(max_similarity),
            "least_similar_pair": least_similar_pair,
            "least_similar_score": float(min_similarity),
            "per_cluster_similarity": nearest_neighbour_summary
        }

    def most_distinct_cluster(self, gmm):
        # Find the cluster that is most separated from all others on average
        centres = gmm.means_
        avg_distances = []

        for i in range(len(centres)):
            distances = []

            for j in range(len(centres)):
                if i != j:
                    distances.append(np.linalg.norm(centres[i] - centres[j]))

            avg_distances.append(np.mean(distances))

        most_distinct = int(np.argmax(avg_distances))

        return {
            "cluster": most_distinct,
            "average_distance_to_other_clusters": float(avg_distances[most_distinct])
        }

    def summarise_clusters(self, X, labels, gmm):
        # Combine all cluster metrics into one DataFrame
        sizes = self.cluster_sizes(labels)
        spreads = self.average_distance_to_centroid(X, labels, gmm)
        centres = self.cluster_centres_summary(gmm)
        feature_summaries = self.feature_importance(X, gmm)
        nearest_info = self.nearest_cluster(gmm)

        similarity_info = self.similarity_summary(gmm)["per_cluster_similarity"]
        similarity_lookup = {item["cluster"]: item for item in similarity_info}

        summary = []

        for centre_info in centres:
            cluster_id = centre_info["cluster"]
            diff = feature_summaries[cluster_id]

            summary.append({
                "cluster": cluster_id,
                "size": sizes.get(cluster_id, 0),
                "average_distance_to_centroid": spreads.get(cluster_id, 0.0),
                "centre": centre_info["centre"],
                "feature_difference": diff.tolist(),
                "feature_description": self.describe_features(diff),
                "nearest_cluster": nearest_info[cluster_id]["nearest_cluster"],
                "nearest_cluster_distance": nearest_info[cluster_id]["distance"],
                "most_similar_cluster": similarity_lookup[cluster_id]["most_similar_cluster"],
                "similarity_score": similarity_lookup[cluster_id]["similarity_score"]
            })

        return pd.DataFrame(summary)

    def generate_text_summary(self, df):
        # Turn the structured cluster summary into readable report sentences
        summaries = []

        for _, row in df.iterrows():
            cluster = int(row["cluster"])
            size = int(row["size"])
            spread = float(row["average_distance_to_centroid"])
            feature_desc = row["feature_description"]
            nearest_cluster = int(row["nearest_cluster"])
            nearest_distance = float(row["nearest_cluster_distance"])
            similar_cluster = int(row["most_similar_cluster"])
            similarity_score = float(row["similarity_score"])

            # Describe compactness
            if spread < 0.2:
                spread_desc = "very compact"
            elif spread < 0.5:
                spread_desc = "moderately compact"
            else:
                spread_desc = "spread out"

            # Describe size
            if size > 5000:
                size_desc = "a large cluster"
            elif size > 2000:
                size_desc = "a medium-sized cluster"
            else:
                size_desc = "a small cluster"

            # Describe similarity strength
            if similarity_score > 0.6:
                sim_desc = f"It is highly similar to cluster {similar_cluster}."
            elif similarity_score > 0.4:
                sim_desc = f"It has moderate similarity to cluster {similar_cluster}."
            else:
                sim_desc = f"It is relatively distinct, with cluster {similar_cluster} as its closest similarity match."

            summaries.append(
                f"Cluster {cluster} is {size_desc} with {size} points and is {spread_desc}. "
                f"It stands out through {feature_desc}. "
                f"Its nearest cluster in latent space is cluster {nearest_cluster} "
                f"(distance {nearest_distance:.3f}). {sim_desc}"
            )

        return summaries

    def find_extreme_clusters(self, df):
        # Find the largest, smallest, most compact, and most spread out clusters
        largest_cluster = df.loc[df["size"].idxmax()]
        smallest_cluster = df.loc[df["size"].idxmin()]
        most_compact_cluster = df.loc[df["average_distance_to_centroid"].idxmin()]
        most_spread_cluster = df.loc[df["average_distance_to_centroid"].idxmax()]

        return {
            "largest_cluster": int(largest_cluster["cluster"]),
            "largest_size": int(largest_cluster["size"]),
            "smallest_cluster": int(smallest_cluster["cluster"]),
            "smallest_size": int(smallest_cluster["size"]),
            "most_compact_cluster": int(most_compact_cluster["cluster"]),
            "most_compact_distance": float(most_compact_cluster["average_distance_to_centroid"]),
            "most_spread_cluster": int(most_spread_cluster["cluster"]),
            "most_spread_distance": float(most_spread_cluster["average_distance_to_centroid"])
        }

    def generate_overall_summary(self, df, gmm):
        # Generate overall summary statements for the full report
        extremes = self.find_extreme_clusters(df)
        distinct_info = self.most_distinct_cluster(gmm)
        sim_info = self.similarity_summary(gmm)

        pair_a, pair_b = sim_info["most_similar_pair"]
        far_a, far_b = sim_info["least_similar_pair"]

        overall_summary = [
            f"Cluster {extremes['largest_cluster']} is the largest cluster with {extremes['largest_size']} points.",
            f"Cluster {extremes['smallest_cluster']} is the smallest cluster with {extremes['smallest_size']} points.",
            f"Cluster {extremes['most_compact_cluster']} is the most compact cluster with an average centroid distance of {extremes['most_compact_distance']:.4f}.",
            f"Cluster {extremes['most_spread_cluster']} is the most spread out cluster with an average centroid distance of {extremes['most_spread_distance']:.4f}.",
            f"Cluster {distinct_info['cluster']} is the most distinct overall, with the greatest average separation from the other cluster centres ({distinct_info['average_distance_to_other_clusters']:.4f}).",
            f"The most similar cluster pair is cluster {pair_a} and cluster {pair_b}, with a similarity score of {sim_info['most_similar_score']:.4f}.",
            f"The least similar cluster pair is cluster {far_a} and cluster {far_b}, with a similarity score of {sim_info['least_similar_score']:.4f}."
        ]

        return overall_summary