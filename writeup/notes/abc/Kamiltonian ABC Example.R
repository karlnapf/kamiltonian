# log-Normal Example

# Simulations
mu0 = 0; t0 = 1/100; t = 1; mu = 2; n = 100; e = 0.1
set.seed(1234)
y = rlnorm(n, mu, t)

# Gaussian moment approximations
muG = function(mu) { exp( mu + 1/(2*t) )  }
tG = function(mu) { exp(-2*mu - 1/t) / (exp(1/t - 1)) }

# densities
prior = function(mu) { dnorm(mu, mu0, sqrt(1/t0) ) }
likelihood = function(mu) { # Gaussian approximation
  loglik = 0
  for (i in 1:n) {
    loglik = loglik + dnorm(y[i], muG(mu), sqrt(1/tG(mu)) + e^2, log = T)
  }
  return( exp(loglik) )
}
unnormalisedpost = function(mu) { prior(mu) * likelihood(mu) }
Z = integrate(unnormalisedpost, lower = -Inf, upper = Inf)$value
interpost = function(mu) { unnormalisedpost(mu)/Z } # numerical integration takes two steps
Z2 = integrate(interpost, lower = -Inf, upper = Inf)$value

# synthetic posterior
post = function(mu) { interpost(mu)/Z2 }

# true posterior
truth = function(mu) { dnorm(mu, (t0*mu0 + t*sum (log(y)) )/(t0 + n*t), sqrt(1/(t0 + n*t)) ) }

# plots
curve(post, 0,5, lwd = 2, xlab = "mu", ylab = "", main = "Log-Normal Example")
curve(truth, 0,5, col = "blue", lwd = 3, lty = 2, add = T)
legend(2.8,6.5, c("True Posterior", "Synthetic Approximation"), c("blue", "black"))
