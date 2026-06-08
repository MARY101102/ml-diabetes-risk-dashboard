export default function ExplanationList({ title, factors, type }) {
  const isIncreasing = type === "increase";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="text-lg font-semibold text-slate-900">{title}</h3>

      {factors.length === 0 ? (
        <p className="mt-3 text-sm text-slate-500">
          No strong factors found for this section.
        </p>
      ) : (
        <div className="mt-4 space-y-3">
          {factors.map((factor) => (
            <div
              key={`${factor.feature}-${factor.shap_value}`}
              className="rounded-xl bg-slate-50 p-4"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="font-medium text-slate-900">{factor.label}</p>
                  <p className="mt-1 text-sm text-slate-500">
                    Input value: {factor.input_value}
                  </p>
                </div>

                <span
                  className={`rounded-full px-3 py-1 text-xs font-semibold ${
                    isIncreasing
                      ? "bg-red-100 text-red-700"
                      : "bg-green-100 text-green-700"
                  }`}
                >
                  SHAP {factor.shap_value}
                </span>
              </div>

              <p className="mt-3 text-sm text-slate-600">
                {factor.explanation}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}