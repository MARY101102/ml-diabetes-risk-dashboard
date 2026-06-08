import ExplanationList from "./ExplanationList";

function getRiskStyle(riskLabel) {
  const label = riskLabel.toLowerCase();

  if (label.includes("high")) {
    return "bg-red-50 text-red-700 border-red-200";
  }

  if (label.includes("medium")) {
    return "bg-yellow-50 text-yellow-700 border-yellow-200";
  }

  return "bg-green-50 text-green-700 border-green-200";
}

export default function ResultCard({ result }) {
  if (!result) {
    return (
      <div className="rounded-3xl border border-dashed border-blue-200 bg-white p-8 text-center shadow-sm">
        <p className="text-lg font-semibold text-slate-800">
          Prediction result will appear here
        </p>
        <p className="mt-2 text-sm text-slate-500">
          Fill the form and submit to estimate diabetes-related risk.
        </p>
      </div>
    );
  }

  return (
    <section className="space-y-5">
      <div className="rounded-3xl border border-blue-100 bg-white p-6 shadow-sm">
        <p className="text-sm font-medium text-slate-500">
          Estimated Diabetes-Related Risk
        </p>

        <div className="mt-4 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div
            className={`rounded-2xl border px-5 py-4 text-2xl font-bold ${getRiskStyle(
              result.estimated_risk
            )}`}
          >
            {result.estimated_risk}
          </div>

          <div className="rounded-2xl bg-blue-50 px-5 py-4 text-right">
            <p className="text-sm text-blue-700">Model Probability</p>
            <p className="text-3xl font-bold text-blue-900">
              {result.display_probability}
            </p>
          </div>
        </div>

        <div className="mt-5 grid gap-4 md:grid-cols-2">
          <div className="rounded-2xl bg-slate-50 p-4">
            <p className="text-sm text-slate-500">Raw model probability</p>
            <p className="mt-1 text-xl font-semibold text-slate-900">
              {result.raw_model_probability}
            </p>
          </div>

          <div className="rounded-2xl bg-slate-50 p-4">
            <p className="text-sm text-slate-500">Calibration method</p>
            <p className="mt-1 text-xl font-semibold capitalize text-slate-900">
              {result.calibration_method}
            </p>
          </div>
        </div>

        <div className="mt-5 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          {result.disclaimer}
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <ExplanationList
          title="Top Factors Increasing Estimated Risk"
          type="increase"
          factors={result.top_factors_increasing_estimated_risk || []}
        />

        <ExplanationList
          title="Top Factors Reducing Estimated Risk"
          type="reduce"
          factors={result.top_factors_reducing_estimated_risk || []}
        />
      </div>
    </section>
  );
}