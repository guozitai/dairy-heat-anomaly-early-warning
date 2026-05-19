
# =============================================================================
# Ningxia heat-stress early-warning figures — v1_10 final bundle
# Figures 1–5 + Figures S1–S5
# =============================================================================
#
# Output:
#   ${LOCAL_ROOT}/heat_stress_figures/figures_v1_10/
#
# Tables:
#   ${LOCAL_ROOT}/heat_stress_figures/tables_v1_10/
#
# Primary prediction parquet:
#   ${DATA_ROOT}/derived/E1_time_anchored_loso_regression_predictions.parquet
#
# Usage:
#   source("/path/to/render_figures_v1_10_final.R")
#   plot_all_figures()
#
# =============================================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(grid)
  library(patchwork)
})

VERSION <- "v1_10"

BASE_DIR    <- "${LOCAL_ROOT}/heat_stress_figures"
TABLE_DIR   <- file.path(BASE_DIR, paste0("tables_", VERSION))
OUT_DIR     <- file.path(BASE_DIR, paste0("figures_", VERSION))
DERIVED_DIR <- "${DATA_ROOT}/derived"

dir.create(TABLE_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(OUT_DIR,   showWarnings = FALSE, recursive = TRUE)

# Backward-compatible name used by older code blocks
DATA_DIR <- TABLE_DIR

# -----------------------------------------------------------------------------
# Palette / theme / IO
# -----------------------------------------------------------------------------

ks_pal <- c(
  navy   = "#1A3D63",
  blue   = "#4472C4",
  orange = "#DD8452",
  salmon = "#E07A5F",
  dkred  = "#B22222",
  green  = "#55A868",
  teal   = "#3D9DA1",
  gold   = "#C2A14D",
  gray   = "#6C757D",
  ltgray = "#D5D8DC",
  black  = "#000000"
)

# Internal farm_id (from parquet) → anonymized Farm label.
# Note: this map is used as a fallback only. Parquet inputs already include
# the 'farm_label' column directly; this map is invoked only if a downstream
# input is missing the farm_label column.
FARM_MAP <- c(
  "6"  = "Farm 1",
  "17" = "Farm 2",
  "16" = "Farm 3",
  "13" = "Farm 4",
  "14" = "Farm 5",
  "15" = "Farm 6",
  "5"  = "Farm 7",
  "3"  = "Farm 8"
)

theme_cea <- function(base_size = 8.6) {
  theme_classic(base_size = base_size, base_family = "sans") +
    theme(
      text = element_text(family = "sans"),
      plot.title = element_text(size = base_size + 0.6, face = "plain", color = "black"),
      plot.subtitle = element_text(size = base_size - 1.1, face = "plain", color = "grey35", lineheight = 1.12),
      axis.title = element_text(size = base_size - 0.2, face = "plain", color = "black"),
      axis.text = element_text(size = base_size - 0.8, face = "plain", color = "black"),
      axis.line = element_line(color = "black", linewidth = 0.28),
      axis.ticks = element_line(color = "black", linewidth = 0.28),
      legend.title = element_blank(),
      legend.text = element_text(size = base_size - 1.1, color = "black"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(4, 8, 5, 6)
    )
}

save_final <- function(p, stub, width, height) {
  pdf_file  <- file.path(OUT_DIR, paste0(stub, ".pdf"))
  tiff_file <- file.path(OUT_DIR, paste0(stub, ".tiff"))

  ggsave(
    filename = pdf_file,
    plot = p,
    width = width,
    height = height,
    units = "mm",
    device = grDevices::cairo_pdf
  )

  ggsave(
    filename = tiff_file,
    plot = p,
    width = width,
    height = height,
    units = "mm",
    dpi = 600,
    compression = "lzw"
  )

  message("Saved: ", pdf_file)
  message("Saved: ", tiff_file)
}

first_existing <- function(paths, required = TRUE, label = "file") {
  hit <- paths[file.exists(paths)][1]
  if (is.na(hit) || length(hit) == 0) {
    if (required) {
      stop("Cannot find ", label, ". Tried:\n", paste(paths, collapse = "\n"))
    } else {
      return(NA_character_)
    }
  }
  hit
}

table_file <- function(filename, required = TRUE) {
  first_existing(
    c(
      file.path(TABLE_DIR, filename),
      file.path(DERIVED_DIR, filename),
      file.path(BASE_DIR, filename),
      file.path("${DATA_ROOT}/tables_v1_9_3", filename),
      file.path("${DATA_ROOT}/derived", filename)
    ),
    required = required,
    label = filename
  )
}

# -----------------------------------------------------------------------------
# Prediction loader used by Fig 4, Fig 5, Fig S3
# -----------------------------------------------------------------------------

.pred_cache <- NULL

load_predictions <- function() {
  if (!is.null(.pred_cache)) return(.pred_cache)

  pred_csv <- file.path(TABLE_DIR, "predictions_ranked.csv")
  pred_parquet <- "${DATA_ROOT}/derived/E1_time_anchored_loso_regression_predictions.parquet"

  if (file.exists(pred_csv)) {
    pred <- readr::read_csv(pred_csv, show_col_types = FALSE)
  } else if (file.exists(pred_parquet)) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("Package 'arrow' is required to read parquet prediction file. Install arrow or provide predictions_ranked.csv.")
    }
    pred <- arrow::read_parquet(pred_parquet)
  } else {
    stop("Cannot find prediction input. Expected either:\n", pred_csv, "\n", pred_parquet)
  }

  # Farm label
  if (!"farm_label" %in% names(pred)) {
    farm_col <- intersect(c("farm", "farm_id"), names(pred))[1]
    if (is.na(farm_col)) stop("Prediction file lacks farm_label / farm / farm_id.")
    pred$farm_label <- as.character(pred[[farm_col]])
    pred$farm_label <- ifelse(
      pred$farm_label %in% names(FARM_MAP),
      unname(FARM_MAP[pred$farm_label]),
      ifelse(grepl("^Farm", pred$farm_label), pred$farm_label, paste("Farm", pred$farm_label))
    )
  }

  # Event label
  if (!"event" %in% names(pred)) {
    event_col <- intersect(c("label", "y", "y_true", "is_event", "anomaly"), names(pred))[1]
    if (is.na(event_col)) stop("Prediction file lacks event label column.")
    pred$event <- pred[[event_col]]
  }
  pred$event <- as.numeric(pred$event)

  # Ratio / production ratio
  if (!"ratio_clip" %in% names(pred)) {
    ratio_col <- intersect(c("ratio", "actual_ratio", "yield_ratio", "r"), names(pred))[1]
    if (is.na(ratio_col)) {
      warning("No ratio column found; using NA ratio_clip. Density/scatter figures will fail.")
      pred$ratio_clip <- NA_real_
    } else {
      pred$ratio_clip <- pmin(pmax(as.numeric(pred[[ratio_col]]), 0.2), 1.85)
    }
  }

  # Risk percentile / decile
  if (!"risk_pct" %in% names(pred)) {
    score_col <- intersect(
      c("risk_score", "score", "pred_score", "prediction", "pred", "y_pred", "pred_ratio", "predicted_ratio"),
      names(pred)
    )[1]
    if (is.na(score_col)) stop("Prediction file lacks risk_pct and no recognizable risk score column.")

    day_col <- intersect(c("day", "date", "test_date", "milk_date"), names(pred))[1]
    if (is.na(day_col)) {
      warning("No day/date column found; ranking within farm only, not farm-date.")
      pred$.rank_group <- pred$farm_label
    } else {
      pred$.rank_group <- paste(pred$farm_label, pred[[day_col]], sep = "__")
    }

    score <- as.numeric(pred[[score_col]])

    # risk_score convention: higher = higher risk. If column is predicted ratio, lower = higher risk.
    ratio_like <- grepl("ratio", score_col, ignore.case = TRUE)
    pred$.score_for_rank <- if (ratio_like) score else -score

    pred <- pred %>%
      group_by(.rank_group) %>%
      mutate(
        risk_rank = rank(.score_for_rank, ties.method = "first"),
        n_day = n(),
        risk_pct = (risk_rank - 0.5) / n_day * 100
      ) %>%
      ungroup() %>%
      select(-.rank_group, -.score_for_rank)
  }

  if (!"risk_decile" %in% names(pred)) {
    pred <- pred %>%
      mutate(risk_decile = pmin(floor(risk_pct / 10), 9))
  }

  pred <- pred %>%
    mutate(
      farm_label = factor(as.character(farm_label), levels = paste("Farm", 1:8)),
      risk_pct = as.numeric(risk_pct),
      risk_decile = as.integer(risk_decile),
      ratio_clip = as.numeric(ratio_clip)
    )

  overall_prev <- mean(pred$event, na.rm = TRUE)
  ratio_lims <- c(0.2, 1.85)

  .pred_cache <<- list(
    pred_ranked = pred,
    overall_prev = overall_prev,
    ratio_lims = ratio_lims
  )
  .pred_cache
}

# =============================================================================
# MAIN FIGURES
# =============================================================================

# -----------------------------------------------------------------------------
# Figure 1 — Within-cow paired heat-stress response
# -----------------------------------------------------------------------------

plot_fig1 <- function() {
  library(ggrepel)

  paired_path <- table_file("paired_comparison_v2.csv", required = FALSE)
  if (is.na(paired_path)) {
    paired_path <- first_existing(
      c(
        file.path(DERIVED_DIR, "paired_comparison_v2.csv"),
        "${DATA_ROOT}/derived/paired_comparison_v2.csv"
      ),
      label = "paired_comparison_v2.csv"
    )
  }

  paired_raw <- readr::read_csv(paired_path, show_col_types = FALSE)

  prev_lookup <- c(
    "Farm 1" = 2.23, "Farm 2" = 9.52, "Farm 3" = 5.47, "Farm 4" = 6.96,
    "Farm 5" = 14.17, "Farm 6" = 14.72, "Farm 7" = 2.19, "Farm 8" = 1.29
  )

  region_lookup <- c(
    "Farm 1" = "Xixia",   "Farm 2" = "Lingwu",  "Farm 3" = "Pingluo",
    "Farm 4" = "Lingwu",  "Farm 5" = "Lingwu",  "Farm 6" = "Pingluo",
    "Farm 7" = "Xixia",   "Farm 8" = "Xixia"
  )

  paired <- paired_raw %>%
    mutate(
      delta        = median_diff,
      delta_ci_lo  = ci_lo,
      delta_ci_hi  = ci_hi,
      n_pairs_num  = n_cows_paired,
      farm_id_chr  = if ("farm_id" %in% names(.)) as.character(farm_id) else NA_character_,
      Farm         = if ("Farm" %in% names(.)) as.character(Farm) else unname(FARM_MAP[farm_id_chr]),
      Farm         = ifelse(is.na(Farm), paste("Farm", farm_id_chr), Farm),
      prev         = prev_lookup[Farm],
      region       = region_lookup[Farm],
      x_plot = prev + case_when(
        Farm == "Farm 1" ~  0.10,
        Farm == "Farm 7" ~ -0.10,
        Farm == "Farm 5" ~ -0.05,
        Farm == "Farm 6" ~  0.05,
        TRUE ~ 0
      ),
      label = gsub("Farm ", "F", Farm)
    )

  region_colors <- c(
    "Xixia"   = unname(ks_pal["navy"]),
    "Lingwu"  = unname(ks_pal["teal"]),
    "Pingluo" = unname(ks_pal["dkred"])
  )

  p1 <- ggplot(paired, aes(x = x_plot, y = delta, color = region)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey55", linewidth = 0.35) +
    geom_point(aes(size = n_pairs_num), alpha = 0.42, stroke = 0) +
    geom_errorbar(aes(ymin = delta_ci_lo, ymax = delta_ci_hi),
                  width = 0.07, linewidth = 0.85, alpha = 0.95) +
    geom_point(size = 1.15, alpha = 0.95, show.legend = FALSE) +
    ggrepel::geom_text_repel(
      aes(label = label),
      size = 2.45,
      color = "black",
      family = "sans",
      fontface = "plain",
      show.legend = FALSE,
      box.padding = 0.20,
      point.padding = 0.14,
      min.segment.length = 0,
      segment.color = "grey72",
      segment.size = 0.18,
      max.overlaps = Inf,
      seed = 12
    ) +
    scale_color_manual(values = region_colors, name = "County") +
    scale_size_continuous(
      name = "Paired cows (n)",
      range = c(2.1, 4.0),
      breaks = c(2000, 4000, 6000),
      labels = c("2K", "4K", "6K")
    ) +
    scale_x_continuous(
      labels = function(x) paste0(x, "%"),
      breaks = c(0, 4, 8, 12, 16),
      limits = c(-0.3, 16.8),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = c(-3.35, 1.75),
      breaks = c(-3, -2, -1, 0, 1),
      expand = c(0, 0)
    ) +
    labs(
      x = "Anomaly prevalence (%)",
      y = "Paired Δ (kg/cow/day, heat-stress − cool-comfort)"
    ) +
    theme_cea(base_size = 8.6) +
    theme(
      legend.position = "right",
      legend.box = "vertical",
      legend.spacing.y = unit(2.2, "mm"),
      legend.key.height = unit(4.5, "mm"),
      legend.title = element_text(size = 8.0, face = "plain", color = "black"),
      legend.text = element_text(size = 7.5, face = "plain", color = "black"),
      panel.grid.major.y = element_line(color = "grey92", linewidth = 0.25),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(6, 8, 5, 6)
    ) +
    guides(
      color = guide_legend(order = 1, override.aes = list(size = 3.0, alpha = 0.85)),
      size = guide_legend(order = 2, override.aes = list(color = "grey35", alpha = 0.45))
    )

  save_final(p1, "Figure1_paired_delta", width = 145, height = 100)
  invisible(p1)
}

# -----------------------------------------------------------------------------
# Figure 2 — In-farm baseline vs time-anchored LOSO Lift@10%
# -----------------------------------------------------------------------------

plot_fig2 <- function() {
  fig2_path <- table_file("figure2_summary.csv")

  d2 <- readr::read_csv(fig2_path, show_col_types = FALSE) %>%
    mutate(Farm = factor(farm, levels = farm[order(loso_lift10)]))

  p2 <- ggplot(d2, aes(y = Farm)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey60", linewidth = 0.30) +
    geom_segment(aes(x = in_farm_v3_lift10, xend = loso_lift10, yend = Farm),
                 color = "grey86", linewidth = 0.45, lineend = "round") +
    geom_segment(aes(x = loso_ci_lo, xend = loso_ci_hi, yend = Farm),
                 color = unname(ks_pal["orange"]), linewidth = 0.68,
                 alpha = 0.40, lineend = "round") +
    geom_point(aes(x = in_farm_v3_lift10, shape = "In-farm baseline"),
               color = unname(ks_pal["blue"]), size = 2.05, alpha = 0.88) +
    geom_point(aes(x = loso_lift10, shape = "Time-anchored LOSO"),
               color = unname(ks_pal["orange"]), size = 2.45, alpha = 0.95) +
    scale_shape_manual(
      name = NULL,
      values = c("In-farm baseline" = 16, "Time-anchored LOSO" = 17)
    ) +
    scale_x_continuous(limits = c(0.85, 7.35), breaks = seq(1, 7, by = 1), expand = c(0, 0)) +
    labs(x = "Lift@10%", y = NULL) +
    theme_cea(base_size = 8.6) +
    theme(
      axis.title.x = element_text(size = 8.5, face = "plain", color = "black"),
      axis.text.x = element_text(size = 7.8, color = "black"),
      axis.text.y = element_text(size = 7.8, color = "black"),
      legend.position = "top",
      legend.justification = "left",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.margin = margin(0, 0, 1, 0),
      legend.key.width = unit(5.0, "mm"),
      legend.key.height = unit(3.5, "mm"),
      legend.text = element_text(size = 7.4, color = "black"),
      panel.grid.major.x = element_line(color = "grey93", linewidth = 0.25),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(3, 8, 5, 6)
    ) +
    guides(
      shape = guide_legend(
        override.aes = list(size = c(2.05, 2.45), alpha = c(0.88, 0.95))
      )
    )

  save_final(p2, "Figure2_transfer_loso", width = 100, height = 90)
  invisible(p2)
}

# -----------------------------------------------------------------------------
# Figure 3 — Prospective-only vs full-feature agreement
# -----------------------------------------------------------------------------

plot_fig3 <- function() {
  library(ggrepel)

  fig3_path <- table_file("E4_prospective_vs_full_comparison_wide.csv")

  d3 <- readr::read_csv(fig3_path, show_col_types = FALSE) %>%
    transmute(
      Farm = farm,
      farm_num = gsub("Farm ", "", farm),
      full_pr_lift = PR_Lift__full_same_day_THI,
      prosp_pr_lift = PR_Lift__prospective_prior_only,
      full_recall10 = Recall_at_10__full_same_day_THI,
      prosp_recall10 = Recall_at_10__prospective_prior_only
    )

  make_panel <- function(dat, xcol, ycol, panel_title, lim, breaks) {
    dd <- dat %>%
      transmute(
        Farm,
        farm_num,
        full = .data[[xcol]],
        prospective = .data[[ycol]],
        retention = prospective / full
      ) %>%
      mutate(label = ifelse(farm_num %in% c("1", "7", "8"), farm_num, ""))

    mean_ret <- mean(dd$retention, na.rm = TRUE)

    ggplot(dd, aes(x = full, y = prospective)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                  color = "grey62", linewidth = 0.32) +
      geom_point(color = unname(ks_pal["navy"]), fill = unname(ks_pal["navy"]),
                 size = 2.05, alpha = 0.88, shape = 21, stroke = 0.25) +
      ggrepel::geom_text_repel(
        aes(label = label),
        size = 2.05,
        family = "sans",
        color = "grey25",
        min.segment.length = Inf,
        box.padding = 0.18,
        point.padding = 0.12,
        max.overlaps = Inf,
        seed = 13,
        show.legend = FALSE
      ) +
      annotate(
        "text",
        x = lim[1] + 0.04 * diff(lim),
        y = lim[2] - 0.07 * diff(lim),
        label = sprintf("mean retention = %.3f", mean_ret),
        hjust = 0,
        vjust = 1,
        size = 2.15,
        family = "sans",
        color = "grey38"
      ) +
      scale_x_continuous(limits = lim, breaks = breaks, expand = c(0, 0)) +
      scale_y_continuous(limits = lim, breaks = breaks, expand = c(0, 0)) +
      coord_fixed(ratio = 1, clip = "off") +
      labs(title = panel_title, x = "Full feature set", y = "Prospective-only") +
      theme_cea(base_size = 8.6) +
      theme(
        plot.title = element_text(size = 8.5, face = "plain", color = "black"),
        axis.title = element_text(size = 8.2, face = "plain", color = "black"),
        axis.text = element_text(size = 7.4, color = "black"),
        panel.grid.major = element_line(color = "grey93", linewidth = 0.25),
        panel.grid.minor = element_blank(),
        plot.margin = margin(3, 4, 4, 4)
      )
  }

  p_pr <- make_panel(
    d3,
    xcol = "full_pr_lift",
    ycol = "prosp_pr_lift",
    panel_title = "PR-Lift",
    lim = c(0, 22),
    breaks = c(0, 5, 10, 15, 20)
  )

  p_recall <- make_panel(
    d3,
    xcol = "full_recall10",
    ycol = "prosp_recall10",
    panel_title = "Recall@10%",
    lim = c(0, 0.80),
    breaks = c(0, 0.2, 0.4, 0.6, 0.8)
  )

  p3 <- p_pr + p_recall + plot_layout(nrow = 1)

  save_final(p3, "Figure3_prospective_agreement", width = 160, height = 80)
  invisible(p3)
}

# -----------------------------------------------------------------------------
# Figure 4 — Risk-decile enrichment
# -----------------------------------------------------------------------------

plot_fig4 <- function() {
  pr <- load_predictions()
  pred_ranked <- pr$pred_ranked
  overall_prev <- pr$overall_prev

  decile_farm <- pred_ranked %>%
    group_by(farm_label, risk_decile) %>%
    summarise(n = n(), event_rate = mean(event), .groups = "drop") %>%
    mutate(decile_mid = risk_decile * 10 + 5)

  decile_pool <- pred_ranked %>%
    group_by(risk_decile) %>%
    summarise(n = n(), event_rate = mean(event), .groups = "drop") %>%
    mutate(decile_mid = risk_decile * 10 + 5)

  top10_rate <- decile_pool %>% filter(risk_decile == 0) %>% pull(event_rate)
  top10_lift <- top10_rate / overall_prev

  p4 <- ggplot() +
    geom_hline(yintercept = overall_prev, linetype = "dashed",
               color = ks_pal[["gray"]], linewidth = 0.30) +
    geom_line(data = decile_farm,
              aes(x = decile_mid, y = event_rate, group = farm_label),
              color = "grey78", linewidth = 0.38, alpha = 0.85) +
    geom_point(data = decile_farm,
               aes(x = decile_mid, y = event_rate),
               color = "grey70", size = 0.80, alpha = 0.70) +
    geom_line(data = decile_pool,
              aes(x = decile_mid, y = event_rate),
              color = ks_pal[["dkred"]], linewidth = 0.78) +
    geom_point(data = decile_pool,
               aes(x = decile_mid, y = event_rate),
               color = ks_pal[["dkred"]], fill = "white",
               shape = 21, size = 1.90, stroke = 0.42) +
    annotate(
      "text",
      x = 10,
      y = max(decile_pool$event_rate, na.rm = TRUE) * 0.96,
      label = paste0("Top 10%\n", sprintf("%.1f", top10_lift), "× baseline"),
      hjust = 0,
      vjust = 1,
      size = 1.95,
      color = ks_pal[["dkred"]],
      family = "sans"
    ) +
    scale_x_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, by = 10),
      labels = seq(0, 100, by = 10),
      expand = expansion(mult = c(0.02, 0.03))
    ) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      expand = expansion(mult = c(0.02, 0.12))
    ) +
    labs(
      x = "Predicted risk percentile within farm-date\n(0 = highest risk)",
      y = "Observed anomaly rate"
    ) +
    theme_cea(base_size = 7.2) +
    theme(
      axis.title = element_text(size = 7.6, face = "plain", color = "black"),
      axis.text = element_text(size = 6.8, color = "black"),
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      panel.grid.major.y = element_line(color = "grey93", linewidth = 0.25),
      panel.grid.major.x = element_line(color = "grey95", linewidth = 0.20),
      panel.grid.minor = element_blank(),
      plot.margin = margin(3, 8, 5, 6)
    )

  save_final(p4, "Figure4_risk_decile_enrichment", width = 100, height = 50)
  invisible(p4)
}

# -----------------------------------------------------------------------------
# Figure 5 — Farm-specific cow-day risk-ranking patterns
# -----------------------------------------------------------------------------

plot_fig5 <- function() {
  pr <- load_predictions()
  pred_ranked <- pr$pred_ranked
  ratio_lims <- pr$ratio_lims

  has_ggrastr <- requireNamespace("ggrastr", quietly = TRUE)

  MAX_POINTS_PER_FARM <- 45000
  FIG_ALPHA <- 0.15
  FIG_POINT_SIZE <- 0.075

  set.seed(20260514)

  scatter_df <- pred_ranked %>%
    mutate(farm_label = factor(farm_label, levels = paste("Farm", 1:8))) %>%
    group_by(farm_label) %>%
    mutate(.rand = runif(n())) %>%
    arrange(.rand, .by_group = TRUE) %>%
    slice_head(n = MAX_POINTS_PER_FARM) %>%
    ungroup() %>%
    select(-.rand)

  curve_df <- pred_ranked %>%
    mutate(
      farm_label = factor(farm_label, levels = paste("Farm", 1:8)),
      pct_bin = pmin(floor(risk_pct / 5), 19),
      pct_mid = pct_bin * 5 + 2.5
    ) %>%
    group_by(farm_label, pct_bin, pct_mid) %>%
    summarise(med_ratio = median(ratio_clip, na.rm = TRUE), .groups = "drop")

  top10_labels <- pred_ranked %>%
    mutate(farm_label = factor(farm_label, levels = paste("Farm", 1:8))) %>%
    group_by(farm_label) %>%
    summarise(
      base_rate  = mean(event, na.rm = TRUE),
      top10_rate = mean(event[risk_pct < 10], na.rm = TRUE),
      enrich     = top10_rate / base_rate,
      .groups = "drop"
    ) %>%
    mutate(
      label = paste0("Top 10%: ", percent(top10_rate, accuracy = 1),
                     "\n", sprintf("%.1f", enrich), "× baseline"),
      x = 57,
      y = ratio_lims[1] + 0.90 * diff(ratio_lims)
    )

  point_layer <- function(data) {
    if (has_ggrastr) {
      ggrastr::geom_point_rast(
        data = data,
        aes(x = risk_pct, y = ratio_clip),
        color = ks_pal[["navy"]],
        alpha = FIG_ALPHA,
        size = FIG_POINT_SIZE,
        stroke = 0,
        raster.dpi = 600
      )
    } else {
      geom_point(
        data = data,
        aes(x = risk_pct, y = ratio_clip),
        color = ks_pal[["navy"]],
        alpha = FIG_ALPHA,
        size = FIG_POINT_SIZE,
        stroke = 0
      )
    }
  }

  p5 <- ggplot() +
    annotate("rect", xmin = 0, xmax = 10, ymin = -Inf, ymax = Inf,
             fill = ks_pal[["orange"]], alpha = 0.13) +
    point_layer(scatter_df) +
    geom_line(data = curve_df,
              aes(x = pct_mid, y = med_ratio),
              color = ks_pal[["navy"]], linewidth = 0.55) +
    geom_vline(xintercept = 10, linetype = "dashed",
               color = ks_pal[["dkred"]], linewidth = 0.26) +
    geom_text(
      data = top10_labels,
      aes(x = x, y = y, label = label),
      color = ks_pal[["dkred"]],
      size = 2.05,
      family = "sans",
      fontface = "plain",
      hjust = 0,
      vjust = 1,
      lineheight = 0.92
    ) +
    facet_wrap(~ farm_label, ncol = 4) +
    scale_x_continuous(
      limits = c(0, 100),
      breaks = c(0, 10, 50, 100),
      labels = c("0%", "10%", "50%", "100%"),
      expand = c(0, 0)
    ) +
    scale_y_continuous(limits = ratio_lims, breaks = c(0.5, 1.0, 1.5), expand = expansion(mult = c(0.01, 0.01))) +
    labs(
      x = "Predicted risk percentile within farm-date\n(0% = highest risk)",
      y = "Observed production ratio"
    ) +
    theme_cea(base_size = 7.8) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 7.8, face = "plain", color = "black"),
      axis.title = element_text(size = 7.8, face = "plain", color = "black"),
      axis.text = element_text(size = 6.9, color = "black"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.spacing = unit(3.6, "mm"),
      plot.margin = margin(4, 8, 5, 6)
    )

  save_final(p5, "Figure5_farm_specific_scatter", width = 180, height = 122)
  invisible(p5)
}

# =============================================================================
# SUPPLEMENTARY FIGURES
# =============================================================================

# -----------------------------------------------------------------------------
# Figure S1 — Three-way performance decomposition
# -----------------------------------------------------------------------------

get_decomp_data <- function() {
  if (exists("v1_9_3_data", envir = .GlobalEnv)) {
    return(get("v1_9_3_data", envir = .GlobalEnv))
  }

  path <- first_existing(
    c(
      file.path(TABLE_DIR, "decomposition_summary.csv"),
      file.path(TABLE_DIR, "figureS1_decomposition.csv"),
      file.path(TABLE_DIR, "figure3_decomposition.csv"),
      file.path(BASE_DIR, "decomposition_summary.csv")
    ),
    required = FALSE,
    label = "decomposition data"
  )

  if (!is.na(path)) {
    return(readr::read_csv(path, show_col_types = FALSE))
  }

  stop(
    "Decomposition data not found. Provide global v1_9_3_data or ",
    "TABLE_DIR/decomposition_summary.csv with columns: Farm, v3_lift10, v31_lift10, loso_lift10, delta_transfer."
  )
}

plot_suppS1 <- function() {
  d3 <- get_decomp_data() %>%
    mutate(Farm = as.character(Farm)) %>%
    arrange(loso_lift10) %>%
    mutate(Farm = factor(Farm, levels = Farm))

  seg_cohort <- d3 %>%
    transmute(Farm, x = v3_lift10, xend = v31_lift10, Step = "Cohort-baseline step")

  seg_transfer <- d3 %>%
    transmute(
      Farm,
      x = v31_lift10,
      xend = loso_lift10,
      Step = ifelse(delta_transfer >= 0, "Transfer step (+)", "Transfer step (-)")
    )

  point_df <- bind_rows(
    d3 %>% transmute(Farm, x = v3_lift10, Model = "In-farm baseline"),
    d3 %>% transmute(Farm, x = v31_lift10, Model = "Cohort baseline"),
    d3 %>% transmute(Farm, x = loso_lift10, Model = "Time-anchored LOSO")
  ) %>%
    mutate(
      Model = factor(
        Model,
        levels = c("In-farm baseline", "Cohort baseline", "Time-anchored LOSO")
      )
    )

  x_max <- max(c(d3$v3_lift10, d3$v31_lift10, d3$loso_lift10), na.rm = TRUE)

  pS1 <- ggplot() +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey60", linewidth = 0.30) +
    geom_segment(data = seg_cohort,
                 aes(x = x, xend = xend, y = Farm, yend = Farm, color = Step),
                 linewidth = 0.62, alpha = 0.82, lineend = "round") +
    geom_segment(data = seg_transfer,
                 aes(x = x, xend = xend, y = Farm, yend = Farm, color = Step),
                 linewidth = 0.62, alpha = 0.82, lineend = "round") +
    geom_point(data = point_df,
               aes(x = x, y = Farm, shape = Model, fill = Model),
               color = "black", size = 2.05, stroke = 0.34, alpha = 0.95) +
    scale_color_manual(
      name = NULL,
      values = c(
        "Cohort-baseline step" = unname(ks_pal["blue"]),
        "Transfer step (-)" = unname(ks_pal["dkred"]),
        "Transfer step (+)" = unname(ks_pal["green"])
      )
    ) +
    scale_shape_manual(
      name = NULL,
      values = c("In-farm baseline" = 21, "Cohort baseline" = 22, "Time-anchored LOSO" = 21)
    ) +
    scale_fill_manual(
      name = NULL,
      values = c("In-farm baseline" = "grey82", "Cohort baseline" = unname(ks_pal["blue"]), "Time-anchored LOSO" = "white")
    ) +
    scale_x_continuous(limits = c(0.85, x_max + 0.25),
                       breaks = seq(1, ceiling(x_max), by = 1),
                       expand = c(0, 0)) +
    labs(x = "Lift@10%", y = NULL) +
    theme_cea(base_size = 8.0) +
    theme(
      axis.title.x = element_text(size = 8.0, face = "plain", color = "black"),
      axis.text.x = element_text(size = 7.2, color = "black"),
      axis.text.y = element_text(size = 7.2, color = "black"),
      legend.position = "right",
      legend.justification = "center",
      legend.direction = "vertical",
      legend.box = "vertical",
      legend.spacing.y = unit(1.4, "mm"),
      legend.margin = margin(0, 0, 0, 2),
      legend.key.width = unit(4.0, "mm"),
      legend.key.height = unit(3.2, "mm"),
      legend.text = element_text(size = 6.7, color = "black"),
      panel.grid.major.x = element_line(color = "grey93", linewidth = 0.25),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(4, 6, 5, 6)
    ) +
    guides(
      color = guide_legend(order = 1, ncol = 1, override.aes = list(linewidth = 0.70, alpha = 0.85)),
      shape = guide_legend(order = 2, ncol = 1, override.aes = list(size = c(1.9, 1.9, 1.9), alpha = 0.95)),
      fill = "none"
    )

  save_final(pS1, "FigureS1_decomposition", width = 145, height = 85)
  invisible(pS1)
}

# -----------------------------------------------------------------------------
# Figure S2 — Top-K recall curves
# -----------------------------------------------------------------------------

plot_suppS2 <- function() {
  topk_path <- table_file("figure4_panelA_topk.csv")

  topk_raw <- readr::read_csv(topk_path, show_col_types = FALSE)

  topk <- topk_raw %>%
    mutate(
      farm_chr = as.character(if ("farm_id" %in% names(.)) farm_id else farm),
      Farm = unname(FARM_MAP[farm_chr]),
      Farm = ifelse(is.na(Farm), paste0("Farm ", farm_chr), Farm),
      Mode = factor(mode, levels = c("in_farm", "loso"),
                    labels = c("In-farm baseline", "Time-anchored LOSO"))
    ) %>%
    pivot_longer(
      cols = matches("^recall@"),
      names_to = "k_label",
      values_to = "recall"
    ) %>%
    mutate(
      k_pct = readr::parse_number(k_label),
      k_prop = k_pct / 100,
      Farm = factor(Farm, levels = paste("Farm", 1:8))
    ) %>%
    filter(!is.na(Farm), !is.na(Mode), !is.na(recall))

  topk_mean <- topk %>%
    group_by(Mode, k_prop) %>%
    summarise(mean_recall = mean(recall, na.rm = TRUE), .groups = "drop")

  mode_cols <- c(
    "In-farm baseline" = unname(ks_pal["blue"]),
    "Time-anchored LOSO" = unname(ks_pal["orange"])
  )

  pS2 <- ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted",
                color = "grey60", linewidth = 0.30) +
    geom_line(data = topk,
              aes(x = k_prop, y = recall, group = interaction(Farm, Mode), color = Mode),
              linewidth = 0.34, alpha = 0.24) +
    geom_point(data = topk,
               aes(x = k_prop, y = recall, color = Mode),
               size = 0.65, alpha = 0.24) +
    geom_line(data = topk_mean,
              aes(x = k_prop, y = mean_recall, color = Mode),
              linewidth = 0.82, alpha = 0.95) +
    geom_point(data = topk_mean,
               aes(x = k_prop, y = mean_recall, color = Mode),
               fill = "white", shape = 21, size = 1.75, stroke = 0.42, alpha = 0.95) +
    scale_color_manual(name = NULL, values = mode_cols) +
    scale_x_continuous(
      breaks = c(0.01, 0.05, 0.10, 0.20),
      labels = scales::percent_format(accuracy = 1),
      expand = expansion(mult = c(0.04, 0.06))
    ) +
    scale_y_continuous(
      labels = scales::percent_format(accuracy = 1),
      expand = expansion(mult = c(0.02, 0.06))
    ) +
    labs(x = "Daily intervention budget (Top-K% of herd)", y = "Anomaly recall") +
    theme_cea(base_size = 8.0) +
    theme(
      axis.title = element_text(size = 8.0, face = "plain", color = "black"),
      axis.text = element_text(size = 7.2, color = "black"),
      legend.position = "top",
      legend.justification = "left",
      legend.direction = "horizontal",
      legend.margin = margin(0, 0, 1, 0),
      legend.key.width = unit(5.0, "mm"),
      legend.key.height = unit(3.5, "mm"),
      legend.text = element_text(size = 7.3, color = "black"),
      panel.grid.major.y = element_line(color = "grey93", linewidth = 0.25),
      panel.grid.major.x = element_line(color = "grey95", linewidth = 0.20),
      panel.grid.minor = element_blank(),
      plot.margin = margin(4, 8, 5, 6)
    )

  save_final(pS2, "FigureS2_topk_curves", width = 120, height = 80)
  invisible(pS2)
}

# -----------------------------------------------------------------------------
# Figure S3 — Pooled cow-day density by predicted risk percentile
# -----------------------------------------------------------------------------

plot_suppS3 <- function() {
  pr <- load_predictions()
  pred_ranked <- pr$pred_ranked
  ratio_lims <- pr$ratio_lims

  n_x_bins <- 100
  n_y_bins <- 120

  density_df <- pred_ranked %>%
    mutate(
      x_bin = pmin(floor(risk_pct / 100 * n_x_bins), n_x_bins - 1),
      y_bin = pmin(
        floor((ratio_clip - ratio_lims[1]) / (ratio_lims[2] - ratio_lims[1]) * n_y_bins),
        n_y_bins - 1
      ),
      x_mid = (x_bin + 0.5) / n_x_bins * 100,
      y_mid = ratio_lims[1] + (y_bin + 0.5) / n_y_bins * (ratio_lims[2] - ratio_lims[1])
    ) %>%
    count(x_mid, y_mid, name = "n") %>%
    mutate(log_n = log10(n + 1))

  curve_pool <- pred_ranked %>%
    mutate(
      pct_bin = pmin(floor(risk_pct / 5), 19),
      pct_mid = pct_bin * 5 + 2.5
    ) %>%
    group_by(pct_bin, pct_mid) %>%
    summarise(med_ratio = median(ratio_clip, na.rm = TRUE), .groups = "drop")

  pS3 <- ggplot() +
    annotate("rect", xmin = 0, xmax = 10, ymin = -Inf, ymax = Inf,
             fill = ks_pal[["orange"]], alpha = 0.14) +
    geom_tile(data = density_df,
              aes(x = x_mid, y = y_mid, fill = log_n),
              width = 100 / n_x_bins,
              height = (ratio_lims[2] - ratio_lims[1]) / n_y_bins) +
    geom_line(data = curve_pool,
              aes(x = pct_mid, y = med_ratio),
              color = ks_pal[["navy"]], linewidth = 0.70) +
    geom_vline(xintercept = 10, linetype = "dashed",
               color = ks_pal[["dkred"]], linewidth = 0.28) +
    annotate("text",
             x = 12,
             y = ratio_lims[2] - 0.04 * diff(ratio_lims),
             label = "Top 10%\ncutoff",
             color = ks_pal[["dkred"]],
             size = 2.05,
             hjust = 0,
             vjust = 1,
             family = "sans") +
    scale_fill_gradientn(
      colors = c("#F7FBFB", "#C6DBEF", "#6BAED6", "#2171B5", unname(ks_pal["navy"])),
      name = expression(log[10]~"(cow-days + 1)"),
      guide = guide_colorbar(
        ticks.colour = "black",
        frame.colour = "black",
        barheight = unit(24, "mm"),
        barwidth = unit(3.0, "mm")
      )
    ) +
    scale_x_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, by = 20),
      labels = function(x) paste0(x, "%"),
      expand = c(0, 0)
    ) +
    scale_y_continuous(limits = ratio_lims, breaks = c(0.5, 1.0, 1.5), expand = expansion(mult = c(0.01, 0.01))) +
    labs(
      x = "Predicted risk percentile within farm-date\n(0% = highest risk)",
      y = "Observed production ratio"
    ) +
    theme_cea(base_size = 8.0) +
    theme(
      axis.title = element_text(size = 8.0, face = "plain", color = "black"),
      axis.text = element_text(size = 7.3, color = "black"),
      legend.position = "right",
      legend.title = element_text(size = 7.4, color = "black"),
      legend.text = element_text(size = 6.8, color = "black"),
      panel.grid = element_blank(),
      plot.margin = margin(4, 8, 5, 6)
    )

  save_final(pS3, "FigureS3_risk_density", width = 120, height = 80)
  invisible(pS3)
}

# -----------------------------------------------------------------------------
# Figure S4 — Farm-level high-resolution risk-stratification curves
# -----------------------------------------------------------------------------

plot_suppS4 <- function() {
  pooled_path <- table_file("fig5_reliability_pooled.csv")
  farm_path   <- table_file("fig5_reliability_per_farm.csv")

  pooled <- readr::read_csv(pooled_path, show_col_types = FALSE) %>%
    mutate(
      bin_pct = as.numeric(bin_mid) * 100,
      mean_rate = as.numeric(mean_rate)
    )

  per_farm <- readr::read_csv(farm_path, show_col_types = FALSE) %>%
    mutate(
      bin_pct = as.numeric(bin_mid) * 100,
      event_rate = as.numeric(event_rate)
    )

  pS4 <- ggplot() +
    geom_line(data = per_farm,
              aes(x = bin_pct, y = event_rate, group = farm_label),
              color = "grey72", linewidth = 0.32, alpha = 0.55) +
    geom_line(data = pooled,
              aes(x = bin_pct, y = mean_rate),
              color = ks_pal[["dkred"]], linewidth = 0.78, alpha = 0.95) +
    geom_point(data = pooled,
               aes(x = bin_pct, y = mean_rate),
               color = ks_pal[["dkred"]], fill = "white",
               shape = 21, size = 1.35, stroke = 0.34) +
    scale_x_continuous(
      limits = c(0, 100),
      breaks = seq(0, 100, by = 25),
      labels = function(x) paste0(x, "%"),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = c(0, 0.85),
      breaks = seq(0, 0.8, by = 0.2),
      labels = scales::percent_format(accuracy = 1),
      expand = c(0, 0)
    ) +
    labs(
      x = "Predicted risk percentile bin\n(0% = lowest risk; 100% = highest risk)",
      y = "Observed anomaly rate"
    ) +
    theme_cea(base_size = 7.6) +
    theme(
      axis.title = element_text(size = 7.7, face = "plain", color = "black"),
      axis.text = element_text(size = 6.9, color = "black"),
      panel.grid.major = element_line(color = "grey93", linewidth = 0.25),
      panel.grid.minor = element_blank(),
      plot.margin = margin(3, 8, 5, 6)
    )

  save_final(pS4, "FigureS4_farm_level_risk_stratification", width = 100, height = 65)
  invisible(pS4)
}

# -----------------------------------------------------------------------------
# Figure S5 — Common-label sanity analysis
# -----------------------------------------------------------------------------

plot_suppS5 <- function() {
  library(ggrepel)

  s5_path <- table_file("common_label_sanity.csv")

  d_wide <- readr::read_csv(s5_path, show_col_types = FALSE) %>%
    mutate(
      farm = ifelse(grepl("^Farm", as.character(farm)), as.character(farm), paste("Farm", farm)),
      framework = recode(
        fixed_label,
        "cohort_defined" = "Fixed cohort-defined label",
        "cow_history_defined" = "Fixed cow-history-defined label"
      )
    )

  d_seg <- d_wide %>%
    transmute(
      farm,
      framework,
      x = cow_baseline_pr_lift,
      xend = cohort_baseline_pr_lift,
      winner = ifelse(cohort_baseline_pr_lift >= cow_baseline_pr_lift,
                      "Cohort baseline higher", "Cow baseline higher")
    )

  d_long <- bind_rows(
    d_wide %>%
      transmute(farm, framework, Feature = "Cow baseline", PR_Lift = cow_baseline_pr_lift),
    d_wide %>%
      transmute(farm, framework, Feature = "Cohort baseline", PR_Lift = cohort_baseline_pr_lift)
  ) %>%
    mutate(
      Feature = factor(Feature, levels = c("Cow baseline", "Cohort baseline")),
      farm = factor(farm, levels = paste("Farm", c(1, 2, 3, 4, 5, 7, 8)))
    )

  d_seg <- d_seg %>%
    mutate(farm = factor(farm, levels = paste("Farm", c(1, 2, 3, 4, 5, 7, 8))))

  pS5 <- ggplot() +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey60", linewidth = 0.30) +
    geom_segment(
      data = d_seg,
      aes(x = x, xend = xend, y = farm, yend = farm, color = winner),
      linewidth = 0.64,
      alpha = 0.86,
      lineend = "round"
    ) +
    geom_point(
      data = d_long,
      aes(x = PR_Lift, y = farm, shape = Feature, fill = Feature),
      color = "black",
      size = 2.05,
      stroke = 0.34,
      alpha = 0.95
    ) +
    facet_wrap(~ framework, nrow = 1) +
    scale_color_manual(
      name = NULL,
      values = c(
        "Cohort baseline higher" = unname(ks_pal["navy"]),
        "Cow baseline higher" = unname(ks_pal["dkred"])
      )
    ) +
    scale_shape_manual(
      name = NULL,
      values = c("Cow baseline" = 24, "Cohort baseline" = 21)
    ) +
    scale_fill_manual(
      name = NULL,
      values = c("Cow baseline" = "white", "Cohort baseline" = unname(ks_pal["teal"]))
    ) +
    scale_x_continuous(
      limits = c(0.8, 10.5),
      breaks = seq(1, 10, by = 1),
      expand = c(0, 0)
    ) +
    labs(x = "PR-Lift", y = NULL) +
    theme_cea(base_size = 8.0) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(size = 7.8, face = "plain", color = "black"),
      axis.title.x = element_text(size = 8.0, face = "plain", color = "black"),
      axis.text.x = element_text(size = 7.2, color = "black"),
      axis.text.y = element_text(size = 7.2, color = "black"),
      legend.position = "top",
      legend.justification = "left",
      legend.direction = "horizontal",
      legend.box = "vertical",
      legend.spacing.y = unit(1.2, "mm"),
      legend.key.width = unit(5, "mm"),
      legend.key.height = unit(3.5, "mm"),
      legend.text = element_text(size = 6.8, color = "black"),
      panel.grid.major.x = element_line(color = "grey93", linewidth = 0.25),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      panel.spacing.x = unit(7, "mm"),
      plot.margin = margin(4, 8, 5, 6)
    ) +
    guides(
      color = guide_legend(order = 1, nrow = 1),
      shape = guide_legend(order = 2, nrow = 1),
      fill = "none"
    )

  save_final(pS5, "FigureS5_common_label_sanity", width = 170, height = 95)
  invisible(pS5)
}

# =============================================================================
# Entrypoints
# =============================================================================

plot_main_all <- function() {
  plot_fig1()
  plot_fig2()
  plot_fig3()
  plot_fig4()
  plot_fig5()
}

plot_supp_all <- function() {
  plot_suppS1()
  plot_suppS2()
  plot_suppS3()
  plot_suppS4()
  plot_suppS5()
}

plot_all_figures <- function() {
  plot_main_all()
  plot_supp_all()
}

message("Loaded v1_10 final figure script.")
message("Run plot_all_figures() to export Figure1–5 and FigureS1–S5.")
