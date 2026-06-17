
library(readxl)
library(MASS)        # glm.nb()
library(pscl)        # zeroinfl(), vuong()
library(lmtest)      # lrtest(), coeftest()
library(AER)         # dispersiontest()
library(rstanarm)    # Bayesian regression via Stan
library(bayestestR)  # Posterior summaries
library(ggplot2)
library(dplyr)
library(tidyr)

# SECTION 1: Data Loading & Preparation

file_path <- file.choose()   # opens a file browser dialog
raw <- as.data.frame(read_excel(file_path))

# The TLC file has a single column named "TLC" containing count values.
# We assign an observation index as the sole predictor.
colnames(raw) <- "TLC"

df <- data.frame(
  Obs  = seq_len(nrow(raw)),   # observation index (1 … N)
  TLC  = as.numeric(raw$TLC)
)

# Drop any NAs
df <- df[!is.na(df$TLC), ]

# Standardise the index (used as sole predictor in all models)
df$Obs_std <- as.numeric(scale(df$Obs))

cat("=== Data Overview ===\n")
print(head(df))
cat("\nRows:", nrow(df), "\n")
print(summary(df$TLC))
cat("\nMean:    ", round(mean(df$TLC), 3), "\n")
cat("Variance:", round(var(df$TLC),  3), "\n")
cat("Dispersion Ratio (Var/Mean):", round(var(df$TLC) / mean(df$TLC), 2), "\n")
cat("Number of zeros:", sum(df$TLC == 0), "\n")
cat("Minimum count:  ", min(df$TLC), "\n")

# SECTION 2: Exploratory Plots

par(mfrow = c(1, 2))

hist(df$TLC, breaks = 20, col = "steelblue", border = "white",
     main = "Distribution of TLC (Count Data)",
     xlab = "Total Leukocyte Count", ylab = "Frequency")

plot(df$Obs, df$TLC, type = "o", col = "steelblue", pch = 16, cex = 0.5,
     main = "TLC Values by Observation Index",
     xlab = "Observation Index", ylab = "TLC")

par(mfrow = c(1, 1))

# SECTION 3: CLASSICAL ESTIMATION (MLE)
# 3a. Classical Poisson Regression

cat("\n\n=== [CLASSICAL] MODEL 1: Poisson Regression ===\n")

pois_model <- glm(TLC ~ Obs_std,
                  data   = df,
                  family = poisson(link = "log"))

print(summary(pois_model))

cat("\n--- Incidence Rate Ratios (exp(coef)) ---\n")
print(exp(cbind(IRR = coef(pois_model), confint(pois_model))))

cat("\n--- Overdispersion Test (Cameron & Trivedi) ---\n")
print(dispersiontest(pois_model))

cat("\nResidual Deviance:", pois_model$deviance,
    "on", pois_model$df.residual, "df\n")
cat("p-value (Chi-sq GoF):",
    pchisq(pois_model$deviance, pois_model$df.residual, lower.tail = FALSE), "\n")

# 3b. Classical Negative Binomial Regression

cat("\n\n=== [CLASSICAL] MODEL 2: Negative Binomial Regression ===\n")

nb_model <- glm.nb(TLC ~ Obs_std, data = df)

print(summary(nb_model))

cat("\n--- Incidence Rate Ratios (exp(coef)) ---\n")
print(exp(cbind(IRR = coef(nb_model), confint(nb_model))))

cat("\nEstimated Theta (dispersion):", nb_model$theta,
    "(SE:", nb_model$SE.theta, ")\n")

cat("\n--- LR Test: Poisson vs Negative Binomial ---\n")
pois_loglik <- logLik(pois_model)
nb_loglik   <- logLik(nb_model)
LR_stat     <- 2 * (as.numeric(nb_loglik) - as.numeric(pois_loglik))
cat("LR Statistic:", round(LR_stat, 4), "\n")
cat("p-value (boundary, 0.5 * chi2_1):",
    round(0.5 * pchisq(LR_stat, 1, lower.tail = FALSE), 6), "\n")

# 3c. Classical Zero-Inflated Poisson (ZIP)
cat("\n\n=== [CLASSICAL] MODEL 3: Zero-Inflated Poisson (ZIP) ===\n")
cat("Number of zeros in TLC data:", sum(df$TLC == 0), "\n")
cat("NOTE: ZIP is included for methodological completeness.\n",
    "      With no zeros, the zero-inflation parameter is not identifiable;\n",
    "      interpret with caution.\n")

zip_model <- tryCatch(
  zeroinfl(TLC ~ Obs_std | 1,
           data = df,
           dist = "poisson",
           link = "logit"),
  error = function(e) {
    cat("ZIP model failed to converge (expected with zero-free data):", e$message, "\n")
    NULL
  }
)

if (!is.null(zip_model)) print(summary(zip_model))

# SECTION 4: BAYESIAN ESTIMATION (MCMC via Stan)
# 4a. Bayesian Poisson Regression
cat("\n\n=== [BAYESIAN] MODEL 1: Poisson Regression ===\n")
cat("Fitting Bayesian Poisson model...\n")

bayes_pois <- stan_glm(
  TLC ~ Obs_std,
  data            = df,
  family          = poisson(link = "log"),
  prior           = normal(0, 10),
  prior_intercept = normal(0, 10),
  chains          = 4,
  iter            = 5000,
  warmup          = 1000,
  seed            = 123,
  refresh         = 0
)

cat("\n--- Posterior Summary ---\n")
print(summary(bayes_pois, digits = 3))

cat("\n--- 95% Credible Intervals ---\n")
print(posterior_interval(bayes_pois, prob = 0.95))

# 4b. Bayesian Negative Binomial Regression
cat("\n\n=== [BAYESIAN] MODEL 2: Negative Binomial Regression ===\n")
cat("Fitting Bayesian Negative Binomial model...\n")

bayes_nb <- stan_glm(
  TLC ~ Obs_std,
  data            = df,
  family          = neg_binomial_2(link = "log"),
  prior           = normal(0, 10),
  prior_intercept = normal(0, 10),
  chains          = 4,
  iter            = 5000,
  warmup          = 1000,
  seed            = 123,
  refresh         = 0
)

cat("\n--- Posterior Summary ---\n")
print(summary(bayes_nb, digits = 3))

cat("\n--- 95% Credible Intervals ---\n")
print(posterior_interval(bayes_nb, prob = 0.95))

# SECTION 5: DIAGNOSTICS
# 5a. Classical: Fitted vs Observed + Pearson Residuals
par(mfrow = c(2, 2))

plot(df$Obs, df$TLC, pch = 16, col = "gray40", cex = 0.6,
     main = "Classical Poisson: Fitted vs Observed",
     xlab = "Observation Index", ylab = "TLC")
lines(df$Obs, fitted(pois_model), col = "red", lwd = 2)

plot(df$Obs, df$TLC, pch = 16, col = "gray40", cex = 0.6,
     main = "Classical NB: Fitted vs Observed",
     xlab = "Observation Index", ylab = "TLC")
lines(df$Obs, fitted(nb_model), col = "darkgreen", lwd = 2)

plot(fitted(pois_model), residuals(pois_model, type = "pearson"),
     pch = 16, col = "steelblue", cex = 0.6,
     main = "Classical Poisson: Pearson Residuals",
     xlab = "Fitted Values", ylab = "Pearson Residuals")
abline(h = 0, lty = 2, col = "red")

plot(fitted(nb_model), residuals(nb_model, type = "pearson"),
     pch = 16, col = "steelblue", cex = 0.6,
     main = "Classical NB: Pearson Residuals",
     xlab = "Fitted Values", ylab = "Pearson Residuals")
abline(h = 0, lty = 2, col = "red")

par(mfrow = c(1, 1))

cat("\nSum of Squared Pearson Residuals:\n")
cat("  Classical Poisson:", sum(residuals(pois_model, type = "pearson")^2), "\n")
cat("  Classical Neg Bin:", sum(residuals(nb_model,   type = "pearson")^2), "\n")

# 5b. Bayesian: Posterior Predictive Checks

pp_pois <- posterior_predict(bayes_pois, draws = 100)
pp_nb   <- posterior_predict(bayes_nb,   draws = 100)

pp_df <- bind_rows(
  as.data.frame(t(pp_pois)) |>
    pivot_longer(everything(), values_to = "value") |>
    mutate(Model = "Bayesian Poisson"),
  as.data.frame(t(pp_nb)) |>
    pivot_longer(everything(), values_to = "value") |>
    mutate(Model = "Bayesian Neg Binomial")
)

print(
  ggplot() +
    geom_density(data = pp_df,
                 aes(x = value, group = name, color = Model),
                 alpha = 0.05, linewidth = 0.3) +
    geom_density(data = df, aes(x = TLC),
                 color = "black", linewidth = 1.2, linetype = "dashed") +
    facet_wrap(~ Model) +
    scale_color_manual(values = c("Bayesian Poisson"      = "steelblue",
                                  "Bayesian Neg Binomial" = "coral")) +
    coord_cartesian(xlim = c(0, quantile(df$TLC, 0.99) * 1.5)) +
    labs(title    = "Bayesian Posterior Predictive Check — TLC",
         subtitle = "Coloured lines = posterior draws; dashed black = observed TLC",
         x = "TLC", y = "Density") +
    theme_minimal() +
    theme(legend.position = "none")
)

# 5c. Bayesian: Observed vs Posterior Predicted (mean)

pred_bayes_pois <- colMeans(posterior_predict(bayes_pois))
pred_bayes_nb   <- colMeans(posterior_predict(bayes_nb))

pred_df <- data.frame(
  Observed                = df$TLC,
  `Bayesian Poisson`      = pred_bayes_pois,
  `Bayesian Neg Binomial` = pred_bayes_nb,
  check.names             = FALSE
) |>
  pivot_longer(-Observed, names_to = "Model", values_to = "Predicted")

print(
  ggplot(pred_df, aes(x = Observed, y = Predicted, color = Model)) +
    geom_point(alpha = 0.5, size = 0.8) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    facet_wrap(~ Model) +
    scale_color_manual(values = c("Bayesian Poisson"       = "steelblue",
                                  "Bayesian Neg Binomial"  = "coral")) +
    labs(title = "Bayesian: Observed vs Posterior Predicted TLC",
         x = "Observed TLC", y = "Predicted TLC") +
    theme_minimal() +
    theme(legend.position = "none")
)

# 5d. Bayesian: Posterior Distributions (Poisson vs NB)

post_pois_df <- as.data.frame(bayes_pois)[, c("(Intercept)", "Obs_std")] |>
  pivot_longer(everything(), names_to = "Parameter", values_to = "Value") |>
  mutate(Model = "Bayesian Poisson")

post_nb_df <- as.data.frame(bayes_nb)[, c("(Intercept)", "Obs_std")] |>
  pivot_longer(everything(), names_to = "Parameter", values_to = "Value") |>
  mutate(Model = "Bayesian Neg Binomial")

post_all <- bind_rows(post_pois_df, post_nb_df)

print(
  ggplot(post_all, aes(x = Value, fill = Model)) +
    geom_density(alpha = 0.5) +
    facet_wrap(~ Parameter, scales = "free") +
    scale_fill_manual(values = c("Bayesian Poisson"      = "steelblue",
                                 "Bayesian Neg Binomial" = "coral")) +
    labs(title = "Posterior Distributions: Bayesian Poisson vs Neg Binomial — TLC",
         x = "Posterior Value", y = "Density") +
    theme_minimal()
)

# 5e. Bayesian: Forest Plot — Credible Intervals

ci_pois <- posterior_interval(bayes_pois, prob = 0.95) |>
  as.data.frame() |>
  tibble::rownames_to_column("Parameter") |>
  mutate(Mean  = colMeans(as.data.frame(bayes_pois))[Parameter],
         Model = "Bayesian Poisson")

ci_nb <- posterior_interval(bayes_nb, prob = 0.95) |>
  as.data.frame() |>
  tibble::rownames_to_column("Parameter") |>
  mutate(Mean  = colMeans(as.data.frame(bayes_nb))[Parameter],
         Model = "Bayesian Neg Binomial")

ci_all <- bind_rows(ci_pois, ci_nb) |>
  filter(Parameter %in% c("(Intercept)", "Obs_std"))

print(
  ggplot(ci_all, aes(x = Mean, y = Parameter, color = Model,
                     xmin = `2.5%`, xmax = `97.5%`)) +
    geom_pointrange(position = position_dodge(width = 0.4), size = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
    scale_color_manual(values = c("Bayesian Poisson"      = "steelblue",
                                  "Bayesian Neg Binomial" = "coral")) +
    labs(title = "Posterior Means with 95% Credible Intervals — TLC",
         x = "Estimate", y = NULL) +
    theme_minimal()
)

# SECTION 6: MODEL COMPARISON
# 6a. Classical: AIC / BIC / Log-Likelihood table

cat("\n\n=== [CLASSICAL] Model Comparison: AIC / BIC / LogLik ===\n")

get_classical_metrics <- function(name, model) {
  ll <- as.numeric(logLik(model))
  k  <- attr(logLik(model), "df")
  data.frame(
    Model  = name,
    LogLik = round(ll, 3),
    AIC    = round(AIC(model), 3),
    BIC    = round(BIC(model), 3),
    df     = k
  )
}

classical_models <- list(
  "Classical Poisson" = pois_model,
  "Classical NB"      = nb_model
)

classical_table <- do.call(rbind, mapply(get_classical_metrics,
                                         names(classical_models),
                                         classical_models,
                                         SIMPLIFY = FALSE))
rownames(classical_table) <- NULL
print(classical_table)

# 6b. Classical: Coefficient Summary (Obs_std)
cat("\n--- Classical Coefficient Summary (Obs_std) ---\n")
cat(sprintf("%-22s  %10s  %10s  %10s\n", "Model", "Estimate", "Std.Error", "p-value"))
cat(strrep("-", 58), "\n")

for (nm in names(classical_models)) {
  ct  <- coeftest(classical_models[[nm]])
  idx <- which(rownames(ct) == "Obs_std")
  cat(sprintf("%-22s  %10.4f  %10.4f  %10.4f\n",
              nm, ct[idx, 1], ct[idx, 2], ct[idx, 4]))
}

# 6c. Bayesian: LOO-CV Comparison

cat("\n\n=== [BAYESIAN] Model Comparison: LOO-CV ===\n")

loo_pois <- loo(bayes_pois)
loo_nb   <- loo(bayes_nb)

cat("\nBayesian Poisson LOO:\n");      print(loo_pois)
cat("\nBayesian Neg Binomial LOO:\n"); print(loo_nb)
cat("\nDirect LOO Comparison:\n")
print(loo_compare(loo_pois, loo_nb))

# LOO bar chart
loo_df <- data.frame(
  Model = c("Bayesian Poisson", "Bayesian Neg Binomial"),
  ELPD  = c(loo_pois$estimates["elpd_loo", "Estimate"],
            loo_nb$estimates["elpd_loo",   "Estimate"]),
  SE    = c(loo_pois$estimates["elpd_loo", "SE"],
            loo_nb$estimates["elpd_loo",   "SE"])
)

print(
  ggplot(loo_df, aes(x = reorder(Model, ELPD), y = ELPD, fill = Model)) +
    geom_col(width = 0.5, alpha = 0.85) +
    geom_errorbar(aes(ymin = ELPD - SE, ymax = ELPD + SE), width = 0.15) +
    scale_fill_manual(values = c("Bayesian Poisson"      = "steelblue",
                                 "Bayesian Neg Binomial" = "coral")) +
    labs(title    = "Bayesian LOO-CV Model Comparison — TLC",
         subtitle = "Higher ELPD = better predictive fit",
         x = NULL, y = "Expected Log Predictive Density (ELPD)") +
    theme_minimal() +
    theme(legend.position = "none")
)

# 6d. Unified Results Table: Classical + Bayesian (Obs_std)

cat("\n\n=== UNIFIED RESULTS TABLE: Classical vs Bayesian (Obs_std) ===\n")

# Classical rows
classical_rows <- lapply(names(classical_models), function(nm) {
  ct  <- coeftest(classical_models[[nm]])
  idx <- which(rownames(ct) == "Obs_std")
  ci  <- confint(classical_models[[nm]])["Obs_std", ]
  data.frame(
    Framework = "Classical (MLE)",
    Model     = nm,
    Parameter = "Obs_std",
    Estimate  = round(ct[idx, 1], 4),
    SD_SE     = round(ct[idx, 2], 4),
    CI_2.5    = round(ci[1], 4),
    CI_97.5   = round(ci[2], 4),
    stringsAsFactors = FALSE
  )
})

# Bayesian rows
extract_bayes <- function(model, model_name) {
  post <- as.data.frame(model)[, "Obs_std"]
  ci   <- quantile(post, c(0.025, 0.975))
  data.frame(
    Framework = "Bayesian (MCMC)",
    Model     = model_name,
    Parameter = "Obs_std",
    Estimate  = round(mean(post), 4),
    SD_SE     = round(sd(post),   4),
    CI_2.5    = round(ci[1],      4),
    CI_97.5   = round(ci[2],      4),
    stringsAsFactors = FALSE
  )
}

bayesian_rows <- list(
  extract_bayes(bayes_pois, "Bayesian Poisson"),
  extract_bayes(bayes_nb,   "Bayesian NB")
)

unified_table <- do.call(rbind, c(classical_rows, bayesian_rows))
rownames(unified_table) <- NULL
print(unified_table)