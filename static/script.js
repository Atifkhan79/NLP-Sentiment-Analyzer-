const textarea = document.getElementById("review");
const charcount = document.getElementById("charcount");
const analyzeBtn = document.getElementById("analyzeBtn");
const result = document.getElementById("result");
const badge = document.getElementById("sentimentBadge");
const cleanedTokens = document.getElementById("cleanedTokens");
const errorMsg = document.getElementById("errorMsg");

const MAX_LEN = 2000;

textarea.addEventListener("input", () => {
  charcount.textContent = `${textarea.value.length} / ${MAX_LEN}`;
});

textarea.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    analyzeBtn.click();
  }
});

analyzeBtn.addEventListener("click", async () => {
  const review = textarea.value.trim();

  errorMsg.hidden = true;
  result.hidden = true;

  if (!review) {
    errorMsg.textContent = "Type or paste a review first.";
    errorMsg.hidden = false;
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review }),
    });

    const data = await res.json();

    if (!res.ok) {
      errorMsg.textContent = data.error || "Something went wrong. Try again.";
      errorMsg.hidden = false;
      return;
    }

    badge.textContent = data.sentiment;
    badge.className = `badge ${data.sentiment.toLowerCase()}`;
    cleanedTokens.textContent = data.cleaned || data.note || "(no tokens survived cleaning)";
    result.hidden = false;
  } catch (err) {
    errorMsg.textContent = "Could not reach the server. Please try again.";
    errorMsg.hidden = false;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze review";
  }
});
