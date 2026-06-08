export default function Header() {
  return (
    <header className="rounded-3xl bg-gradient-to-r from-blue-700 via-blue-600 to-cyan-500 p-8 text-white shadow-lg">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="mb-3 inline-flex rounded-full bg-white/20 px-4 py-1 text-sm font-medium">
            ML-Powered Healthcare Dashboard
          </p>

          <h1 className="text-3xl font-bold tracking-tight md:text-5xl">
            Diabetes Risk Estimation Dashboard
          </h1>

          <p className="mt-4 max-w-3xl text-blue-50">
            Enter health and lifestyle indicators to estimate the model
            probability of the positive dataset class representing prediabetes
            or diabetes.
          </p>
        </div>

        <div className="rounded-2xl bg-white/15 p-5 backdrop-blur">
          <p className="text-sm text-blue-50">Project Status</p>
          <p className="mt-1 text-2xl font-bold">AI Model Connected</p>
          <p className="mt-2 text-sm text-blue-100">
            FastAPI + React + Random Forest
          </p>
        </div>
      </div>
    </header>
  );
}