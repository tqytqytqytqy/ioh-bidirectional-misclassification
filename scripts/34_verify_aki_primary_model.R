args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(2, 3)) {
  stop(paste(
    "Usage: Rscript 34_verify_aki_primary_model.R",
    "model_frame.csv sequential_and_incremental_results.csv [verification.txt]"
  ))
}

model_frame <- read.csv(args[[1]], stringsAsFactors = FALSE)
published <- read.csv(args[[2]], stringsAsFactors = FALSE, check.names = FALSE)
model_frame$aki <- ifelse(
  model_frame$aki == "True",
  1,
  ifelse(model_frame$aki == "False", 0, NA_real_)
)

cluster_fit <- function(data, formula, term) {
  rownames(data) <- NULL
  data$department <- factor(data$department)
  fit <- glm(
    formula,
    data = data,
    family = poisson(link = "log"),
    control = glm.control(epsilon = 1e-12, maxit = 100)
  )
  used <- as.integer(rownames(model.frame(fit)))
  groups <- data$subjectid[used]
  x <- model.matrix(fit)
  mu <- fitted(fit)
  y <- model.response(model.frame(fit))
  score_by_row <- x * as.numeric(y - mu)
  score_by_subject <- rowsum(score_by_row, groups, reorder = FALSE)
  bread <- solve(crossprod(x, x * as.numeric(mu)))
  meat <- crossprod(score_by_subject)
  n <- nrow(x)
  k <- ncol(x)
  g <- nrow(score_by_subject)
  correction <- (g / (g - 1)) * ((n - 1) / (n - k))
  covariance <- bread %*% meat %*% bread * correction
  coefficient <- unname(coef(fit)[[term]])
  standard_error <- sqrt(diag(covariance))[[term]]
  list(
    n = n,
    events = sum(y),
    subjects = g,
    coefficient = coefficient,
    standard_error = standard_error,
    relative_risk = exp(coefficient),
    ci_low = exp(coefficient - qnorm(0.975) * standard_error),
    ci_high = exp(coefficient + qnorm(0.975) * standard_error)
  )
}

check_fit <- function(label, observed, expected) {
  if (nrow(expected) != 1) {
    stop(paste("Result row not found exactly once:", label))
  }
  checks <- c(
    n_cases = observed$n == expected$n_cases,
    events = observed$events == expected$events,
    coefficient = abs(observed$coefficient - expected$coefficient) < 1e-8,
    standard_error = abs(observed$standard_error - expected$standard_error) < 1e-8,
    relative_risk = abs(observed$relative_risk - expected$relative_risk) < 1e-8,
    ci_low = abs(observed$ci_low - expected$ci_low) < 1e-8,
    ci_high = abs(observed$ci_high - expected$ci_high) < 1e-8
  )
  report <- c(
    paste0("MODEL=", label),
    sprintf(
      "N=%d; events=%d; subjects=%d",
      observed$n, observed$events, observed$subjects
    ),
    sprintf(
      "coefficient=%.12f; cluster_SE=%.12f; RR=%.12f; 95%% CI %.12f to %.12f",
      observed$coefficient,
      observed$standard_error,
      observed$relative_risk,
      observed$ci_low,
      observed$ci_high
    ),
    paste(names(checks), ifelse(checks, "PASS", "FAIL"), sep = "=")
  )
  list(report = report, checks = checks)
}

clinical_variables <- c(
  "aki", "age10", "male", "bmi5", "asa_high", "emop", "preop_htn",
  "preop_dm", "baseline_cr05", "duration_hr", "department", "hidden_twa10",
  "subjectid"
)
clinical_data <- model_frame[
  complete.cases(model_frame[, clinical_variables]),
  clinical_variables
]
clinical <- cluster_fit(
  clinical_data,
  aki ~ age10 + male + bmi5 + asa_high + emop + preop_htn + preop_dm +
    baseline_cr05 + duration_hr + department + hidden_twa10,
  "hidden_twa10"
)
clinical_check <- check_fit(
  "Absolute hidden burden, clinical adjusted",
  clinical,
  published[published$analysis == "Absolute hidden burden, clinical adjusted", ]
)

all_basis <- splines::bs(
  model_frame$log_true_twa,
  df = 3,
  degree = 2,
  intercept = FALSE
)
model_frame$burden_bs1 <- all_basis[, 1]
model_frame$burden_bs2 <- all_basis[, 2]
model_frame$burden_bs3 <- all_basis[, 3]
incremental_variables <- c(
  clinical_variables,
  "log_true_twa", "burden_bs1", "burden_bs2", "burden_bs3"
)
incremental_data <- model_frame[
  complete.cases(model_frame[, incremental_variables]),
  incremental_variables
]
incremental <- cluster_fit(
  incremental_data,
  aki ~ age10 + male + bmi5 + asa_high + emop + preop_htn + preop_dm +
    baseline_cr05 + duration_hr + department + burden_bs1 + burden_bs2 +
    burden_bs3 + hidden_twa10,
  "hidden_twa10"
)
incremental_check <- check_fit(
  "Absolute hidden burden, clinical plus reference burden",
  incremental,
  published[
    published$analysis ==
      "Absolute hidden burden, clinical plus reference burden",
  ]
)

hdr_source <- model_frame[model_frame$true_auc > 0, ]
hdr_basis <- splines::bs(
  hdr_source$log_true_twa,
  df = 3,
  degree = 2,
  intercept = FALSE
)
hdr_source$burden_bs1 <- hdr_basis[, 1]
hdr_source$burden_bs2 <- hdr_basis[, 2]
hdr_source$burden_bs3 <- hdr_basis[, 3]
hdr_variables <- c(
  "aki", "age10", "male", "bmi5", "asa_high", "emop", "preop_htn",
  "preop_dm", "baseline_cr05", "duration_hr", "department", "hdr10",
  "subjectid", "burden_bs1", "burden_bs2", "burden_bs3"
)
hdr_data <- hdr_source[
  complete.cases(hdr_source[, hdr_variables]),
  hdr_variables
]
hdr <- cluster_fit(
  hdr_data,
  aki ~ age10 + male + bmi5 + asa_high + emop + preop_htn + preop_dm +
    baseline_cr05 + duration_hr + department + burden_bs1 + burden_bs2 +
    burden_bs3 + hdr10,
  "hdr10"
)
hdr_check <- check_fit(
  "HDR65 incremental model",
  hdr,
  published[published$analysis == "HDR65 incremental model", ]
)

reports <- c(
  clinical_check$report,
  "",
  incremental_check$report,
  "",
  hdr_check$report
)
cat(reports, sep = "\n")
cat("\n")

if (length(args) == 3) {
  writeLines(reports, args[[3]], useBytes = TRUE)
}

all_checks <- c(
  clinical_check$checks,
  incremental_check$checks,
  hdr_check$checks
)
if (!all(all_checks)) {
  stop("Independent R verification did not match one or more Python models")
}
