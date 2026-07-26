"""
WEEK2 Project - Data Visualization
Exploratory Data Analysis on COVID-19 Dataset

Objective : Practice Matplotlib & Seaborn visualizations
Dataset   : COVID-19 Data (Global) - Our World in Data
Columns   : location, date, total_cases, total_deaths, total_vaccinations

Tasks:
    1. Plot trend lines (cases vs date)
    2. Compare top 5 countries
    3. Create heatmap & scatter plot

Run this script from a folder that contains 'covid_data.csv'
(same columns as above). Charts are saved as PNG files into ./charts/
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
CSV_FILE = "covid_data.csv"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded '{path}' -> {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Drop aggregate rows that aren't real countries. OWID marks continents,
    # income groups, and the 'World' total with an iso_code starting with
    # 'OWID_' instead of a real ISO-3166 country code.
    if "iso_code" in df.columns:
        df = df[~df["iso_code"].astype(str).str.startswith("OWID_")].copy()
        df = df.drop(columns=["iso_code"])

    # Fill missing cumulative values by carrying the last known value
    # forward within each country (typical for cumulative COVID stats)
    df = df.sort_values(["location", "date"])
    for col in ["total_cases", "total_deaths", "total_vaccinations"]:
        df[col] = df.groupby("location")[col].transform(lambda s: s.ffill())
        df[col] = df[col].fillna(0)

    print(f"After cleaning -> {df.shape[0]:,} rows, {df['location'].nunique()} countries\n")
    return df


def plot_global_trend(df: pd.DataFrame) -> None:
    """Task 1: trend line of total cases over time (global sum)."""
    global_trend = df.groupby("date")[["total_cases", "total_deaths"]].sum()

    plt.figure(figsize=(11, 5))
    plt.plot(global_trend.index, global_trend["total_cases"], label="Total cases", linewidth=2)
    plt.plot(global_trend.index, global_trend["total_deaths"], label="Total deaths", linewidth=2)
    plt.title("Global COVID-19 Trend: Cases vs Deaths Over Time")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/1_global_trend.png", dpi=150)
    plt.close()
    print("Saved chart: 1_global_trend.png")


def get_top5_countries(df: pd.DataFrame) -> list:
    latest = df.sort_values("date").groupby("location").tail(1)
    top5 = latest.nlargest(5, "total_cases")["location"].tolist()
    print(f"Top 5 countries by total cases: {top5}\n")
    return top5


def plot_top5_comparison(df: pd.DataFrame, top5: list) -> None:
    """Task 2: compare trend lines for the top 5 countries."""
    subset = df[df["location"].isin(top5)]

    plt.figure(figsize=(11, 5))
    for country in top5:
        c_data = subset[subset["location"] == country]
        plt.plot(c_data["date"], c_data["total_cases"], label=country, linewidth=2)
    plt.title("Top 5 Countries by Total COVID-19 Cases")
    plt.xlabel("Date")
    plt.ylabel("Total Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/2_top5_cases_trend.png", dpi=150)
    plt.close()
    print("Saved chart: 2_top5_cases_trend.png")

    # Bonus: bar chart of latest totals (cases, deaths, vaccinations)
    latest = subset.sort_values("date").groupby("location").tail(1).set_index("location")
    latest = latest.loc[top5, ["total_cases", "total_deaths", "total_vaccinations"]]

    latest.plot(kind="bar", figsize=(10, 5), logy=True)
    plt.title("Top 5 Countries - Latest Totals (log scale)")
    plt.ylabel("Count (log scale)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/3_top5_bar_comparison.png", dpi=150)
    plt.close()
    print("Saved chart: 3_top5_bar_comparison.png")


def plot_heatmap(df: pd.DataFrame, top5: list) -> None:
    """Task 3a: heatmap of correlation between cases/deaths/vaccinations."""
    corr_cols = ["total_cases", "total_deaths", "total_vaccinations"]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Heatmap: Cases, Deaths, Vaccinations")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/4_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved chart: 4_correlation_heatmap.png")

    # Bonus heatmap: monthly total cases per top-5 country
    subset = df[df["location"].isin(top5)].copy()
    subset["month"] = subset["date"].dt.to_period("M").astype(str)
    monthly = subset.groupby(["location", "month"])["total_cases"].max().unstack(0)

    plt.figure(figsize=(12, 5))
    sns.heatmap(monthly.T, cmap="YlOrRd", cbar_kws={"label": "Total cases"})
    plt.title("Monthly Total Cases by Country (Top 5)")
    plt.xlabel("Month")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/5_monthly_heatmap_top5.png", dpi=150)
    plt.close()
    print("Saved chart: 5_monthly_heatmap_top5.png")


def plot_scatter(df: pd.DataFrame) -> None:
    """Task 3b: scatter plot of total cases vs total deaths (latest snapshot per country)."""
    latest = df.sort_values("date").groupby("location").tail(1)
    latest = latest[(latest["total_cases"] > 0) & (latest["total_deaths"] > 0)]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=latest,
        x="total_cases",
        y="total_deaths",
        size="total_vaccinations",
        sizes=(20, 400),
        alpha=0.6,
        legend=False,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Total Cases vs Total Deaths per Country\n(bubble size = total vaccinations)")
    plt.xlabel("Total Cases (log scale)")
    plt.ylabel("Total Deaths (log scale)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/6_cases_vs_deaths_scatter.png", dpi=150)
    plt.close()
    print("Saved chart: 6_cases_vs_deaths_scatter.png")


def main():
    df = load_and_clean(CSV_FILE)
    plot_global_trend(df)
    top5 = get_top5_countries(df)
    plot_top5_comparison(df, top5)
    plot_heatmap(df, top5)
    plot_scatter(df)
    print(f"\nAll charts saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
