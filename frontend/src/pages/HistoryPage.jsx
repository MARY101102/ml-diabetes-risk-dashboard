import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getPredictionHistory } from "../services/api";

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadHistory() {
      try {
        setLoading(true);
        const data = await getPredictionHistory();
        setHistory(data);
      } catch (error) {
        setErrorMessage("Could not load prediction history.");
        console.error(error);
      } finally {
        setLoading(false);
      }
    }

    loadHistory();
  }, []);

  return (
    <main className="min-h-screen bg-slate-50 px-4 py-6 md:px-8">
      <div className="mx-auto max-w-6xl">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-slate-900">
              Prediction History
            </h1>
            <p className="mt-2 text-sm text-slate-500">
              View your previous diabetes risk estimations.
            </p>
          </div>

          <Link
            to="/"
            className="rounded-xl bg-blue-700 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-800"
          >
            Back to Dashboard
          </Link>
        </div>

        {loading && (
          <div className="rounded-2xl bg-white p-6 text-slate-600 shadow-sm">
            Loading history...
          </div>
        )}

        {errorMessage && (
          <div className="rounded-2xl bg-red-50 p-4 text-sm text-red-700">
            {errorMessage}
          </div>
        )}

        {!loading && history.length === 0 && (
          <div className="rounded-2xl bg-white p-6 text-slate-600 shadow-sm">
            No prediction history yet.
          </div>
        )}

        <div className="space-y-4">
          {history.map((item) => (
            <div
              key={item.id}
              className="rounded-2xl border border-blue-100 bg-white p-5 shadow-sm"
            >
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="text-sm text-slate-500">
                    Assessment #{item.id}
                  </p>
                  <h2 className="mt-1 text-xl font-bold text-slate-900">
                    {item.estimated_risk}
                  </h2>
                </div>

                <div className="text-left md:text-right">
                  <p className="text-sm text-slate-500">Model Probability</p>
                  <p className="text-2xl font-bold text-blue-700">
                    {item.display_probability}
                  </p>
                </div>
              </div>

              <div className="mt-4 grid gap-3 md:grid-cols-3">
                <div className="rounded-xl bg-slate-50 p-3">
                  <p className="text-sm text-slate-500">Calibration</p>
                  <p className="font-semibold capitalize">
                    {item.calibration_method}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-50 p-3">
                  <p className="text-sm text-slate-500">Raw Probability</p>
                  <p className="font-semibold">
                    {item.raw_model_probability}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-50 p-3">
                  <p className="text-sm text-slate-500">Created At</p>
                  <p className="font-semibold">
                    {new Date(item.created_at).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}