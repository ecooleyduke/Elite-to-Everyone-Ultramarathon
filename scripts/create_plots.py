import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_df():
    return pd.read_csv("./data/TWO_CENTURIES_OF_UM_RACES.csv", low_memory=False)

def athlete_performance_to_seconds(t):
    try:
        t = t.strip("h")
        parts = list(map(int, t.strip().split(":")))
        h, m, s = parts
        return h * 3600 + m * 60 + s
    except:
        return np.nan
    
def create_top_3_df(df):
    # Change `Athlete performance` into seconds
    df["performance_sec"] = df["Athlete performance"].apply(athlete_performance_to_seconds)

    # Filter NaN rows
    df = df[
        df['performance_sec'].notnull() &
        df['Year of event'].notnull()
    ]

    # Filter the df to only rows from the top 10 most frequent event distances
    top_3_distances = (
        df["Event distance/length"]
        .value_counts()
        .nlargest(3)
        .index
    )
    df_top_3 = df[df["Event distance/length"].isin(top_3_distances)]
    return df_top_3

def plot_performance_over_time_top_3(df_top_3):
    # Group by event distance and year of event then average performance
    grouped = df_top_3.groupby(["Event distance/length", "Year of event"]).agg(
        avg_perf_sec=("performance_sec", "mean"),
    ).reset_index()

    grouped["avg_perf_hour"] = grouped["avg_perf_sec"] / 3600

    # Plot out the data
    plt.figure(figsize=(14, 8))

    for distance in sorted(grouped['Event distance/length'].unique()):
        data = grouped[grouped['Event distance/length'] == distance]
        plt.plot(
            data['Year of event'],
            data['avg_perf_hour'],
            marker='o',
            linewidth=2,
            label=distance
        )

    plt.title("Average Ultrarunning Finish Time Over Time by Distance", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Average Finish Time (hours)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Distance", loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_performance_over_time_top_3_1950(df_top_3):
    df_top_3 = df_top_3[df_top_3["Year of event"] > 1950]
    # Group by event distance and year of event then average performance
    grouped = df_top_3.groupby(["Event distance/length", "Year of event"]).agg(
        avg_perf_sec=("performance_sec", "mean"),
    ).reset_index()

    grouped["avg_perf_hour"] = grouped["avg_perf_sec"] / 3600

    # Plot out the data
    plt.figure(figsize=(14, 8))

    for distance in sorted(grouped['Event distance/length'].unique()):
        data = grouped[grouped['Event distance/length'] == distance]
        plt.plot(
            data['Year of event'],
            data['avg_perf_hour'],
            marker='o',
            linewidth=2,
            label=distance
        )

    plt.title("Average Ultramarathon Finish Time Over Time by Distance", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Average Finish Time (hours)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Distance", loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_finishers_per_year(df_top_3):
    df_top_3 = df_top_3[df_top_3["Year of event"] > 1950]
    # Delete duplicate events
    df_unique = df_top_3.drop_duplicates(subset=["Event name", "Year of event", "Event distance/length"])

    # Group by year and sum finishers
    finishers_by_year_distance = (
        df_unique.groupby(["Year of event", "Event distance/length"])["Event number of finishers"]
        .sum()
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(12, 6))

    # Plot each distance separately
    for distance in sorted(finishers_by_year_distance['Event distance/length'].unique()):
        subset = finishers_by_year_distance[finishers_by_year_distance["Event distance/length"] == distance]
        plt.plot(
            subset["Year of event"],
            subset["Event number of finishers"],
            label=distance,
            marker='o',
            linewidth=2,
        )

    # Add labels, grid, legend
    plt.title("Total Ultrarunning Finishers Per Year by Distance", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Total Finishers", fontsize=12)
    plt.legend(title="Distance")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_time_distribution_by_year(df_top_3):
    df_top_3 = df_top_3[df_top_3["Year of event"] > 1950]
    # Convert to hours for plotting
    df_top_3["Performance_hours"] = df_top_3["performance_sec"] / 3600

    # Filter to only 50 km race distance
    df_50 = df_top_3[df_top_3["Event distance/length"] == "50km"]

    grouped = df_50.groupby("Year of event")["Performance_hours"].apply(list).sort_index()

    # Filter out years with too few finishers (e.g., < 30)
    grouped = grouped[grouped.apply(len) > 30]

    # Extract years and lists of data
    years = grouped.index.tolist()
    data = grouped.tolist()

    # Calculate mean per year
    means = [sum(times) / len(times) for times in data]

    # Extract years and data
    years = grouped.index.tolist()
    data = grouped.tolist()

    # Create the box plot
    plt.figure(figsize=(14, 6))
    plt.boxplot(data, positions=range(len(years)), patch_artist=True, showfliers=False)

    # Add mean line
    plt.plot(range(len(years)), means, color='red', linestyle='--', marker='o', label='Mean Finish Time')

    # Customize plot
    plt.title("Finish Time Distribution by Year for the 50 km race", fontsize=16)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Finish Time (Hours)", fontsize=12)
    tick_positions = [i for i, year in enumerate(years) if year % 5 == 0]
    tick_labels = [years[i] for i in tick_positions]
    plt.xticks(ticks=tick_positions, labels=tick_labels, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_age_distribution(df_top_3):
    df_top_3 = df_top_3[df_top_3["Year of event"] > 1950]
    # Filter out unreasonable ages (e.g., < 10 or > 100)
    df_top_3_age = df_top_3[(df_top_3["Age"] >= 10) & (df_top_3["Age"] <= 100)]

    # Group by year and calculate median and IQR
    grouped = df_top_3_age.groupby("Year of event")["Age"]
    median_age = grouped.median()
    q1 = grouped.quantile(0.25)
    q3 = grouped.quantile(0.75)

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(median_age.index, median_age.values, label='Median Age', color='blue')
    plt.fill_between(median_age.index, q1, q3, color='blue', alpha=0.2, label='IQR (25th-75th percentile)')

    plt.title("Age Distribution of Ultramarathon Finishers Over Time")
    plt.xlabel("Year")
    plt.ylabel("Age")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_country_counts_over_time(df_top_3):
    df_top_3 = df_top_3[df_top_3["Year of event"] > 1950]
    country_counts = df_top_3.groupby("Year of event")["Athlete country"].nunique()

    plt.figure(figsize=(14, 6))
    plt.plot(country_counts.index, country_counts.values, marker='o', color='green')

    plt.title("Number of Unique Countries Represented by Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Countries")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()