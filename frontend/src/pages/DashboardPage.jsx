import { useState } from "react";
import { Link } from "react-router-dom";
import Header from "../components/Header";
import MetricCard from "../components/MetricCard";
import PredictionForm from "../components/PredictionForm";
import ResultCard from "../components/ResultCard";
import { useAuth } from "../context/AuthContext";
import { predictDiabetesRisk } from "../services/api";

export default function DashboardPage() {
  const { user, logout } = useAuth();

  const [predictionResult, setPredictionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  async function handlePredictionSubmit(formData) {
    try {
      setLoading(true);
      setErrorMessage("");

      const result = await predictDiabetesRisk(formData);
      setPredictionResult(result);
    } catch (error) {
      setErrorMessage(
        "Prediction request failed. Make sure you are logged in and the backend is running."
      );
      console.error(error);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-7xl px-4 py-6 md:px-8">
        <div className="mb-5 flex flex-col gap-3 rounded-2xl bg-white p-4 shadow-sm md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm text-slate-500">Logged in as</p>
            <p className="font-semibold text-slate-900">
              {user?.name} · {user?.email}
            </p>
          </div>

          <div className="flex gap-3">
            <Link
              to="/history"
              className="rounded-xl border border-blue-200 px-4 py-2 text-sm font-semibold text-blue-700 hover:bg-blue-50"
            >
              View History
            </Link>

            <button
              onClick={logout}
              className="rounded-xl border border-red-200 px-4 py-2 text-sm font-semibold text-red-700 hover:bg-red-50"
            >
              Logout
            </button>
          </div>
        </div>

        <Header />

        <section className="mt-6 grid gap-4 md:grid-cols-3">
          <MetricCard
            title="Model Type"
            value="Random Forest"
            description="Final selected baseline classification model"
          />

          <MetricCard
            title="Final ROC-AUC"
            value="0.8234"
            description="Final test-set class separation score"
          />

          <MetricCard
            title="Final Recall"
            value="0.7429"
            description="Positive-class detection performance"
          />
        </section>

        <section className="mt-8 grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
          <PredictionForm
            onSubmit={handlePredictionSubmit}
            loading={loading}
          />

          <div className="space-y-5">
            {errorMessage && (
              <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                {errorMessage}
              </div>
            )}

            <ResultCard result={predictionResult} />

            <div className="rounded-2xl border border-blue-100 bg-blue-50 p-5 text-sm text-blue-800">
              <p className="font-semibold">Responsible Use Notice</p>
              <p className="mt-2">
                This dashboard is an educational machine learning project. It
                estimates the model probability of the positive dataset class
                representing prediabetes or diabetes. It does not provide
                medical diagnosis, treatment guidance, or medical advice.
              </p>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}