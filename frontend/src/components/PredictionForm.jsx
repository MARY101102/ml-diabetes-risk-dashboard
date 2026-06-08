import { useState } from "react";
import {
  ageOptions,
  binaryOptions,
  generalHealthOptions,
} from "../data/formOptions";

const initialFormData = {
  HighBP: 1,
  HighChol: 1,
  CholCheck: 1,
  BMI: 32,
  Smoker: 0,
  Stroke: 0,
  HeartDiseaseorAttack: 0,
  PhysActivity: 1,
  Fruits: 1,
  Veggies: 1,
  HvyAlcoholConsump: 0,
  AnyHealthcare: 1,
  NoDocbcCost: 0,
  GenHlth: 3,
  MentHlth: 5,
  PhysHlth: 3,
  DiffWalk: 0,
  Age: 8,
};

function SelectField({ label, name, value, options, onChange }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <select
        name={name}
        value={value}
        onChange={onChange}
        className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumberField({ label, name, value, min, max, onChange }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-slate-700">{label}</span>
      <input
        type="number"
        name={name}
        value={value}
        min={min}
        max={max}
        onChange={onChange}
        className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
      />
    </label>
  );
}

export default function PredictionForm({ onSubmit, loading }) {
  const [formData, setFormData] = useState(initialFormData);

  function handleChange(event) {
    const { name, value } = event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: Number(value),
    }));
  }

  function handleSubmit(event) {
    event.preventDefault();
    onSubmit(formData);
  }

  function handleReset() {
    setFormData(initialFormData);
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-3xl border border-blue-100 bg-white p-6 shadow-sm"
    >
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900">
          Diabetes Risk Input Form
        </h2>
        <p className="mt-2 text-sm text-slate-500">
          Fill the health indicator fields below. The result is an educational
          ML-based estimate, not a diagnosis.
        </p>
      </div>

      <div className="grid gap-5 md:grid-cols-2">
        <SelectField
          label="High Blood Pressure"
          name="HighBP"
          value={formData.HighBP}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="High Cholesterol"
          name="HighChol"
          value={formData.HighChol}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Cholesterol Check in Last 5 Years"
          name="CholCheck"
          value={formData.CholCheck}
          options={binaryOptions}
          onChange={handleChange}
        />

        <NumberField
          label="BMI"
          name="BMI"
          value={formData.BMI}
          min={1}
          max={100}
          onChange={handleChange}
        />

        <SelectField
          label="Smoking History"
          name="Smoker"
          value={formData.Smoker}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Stroke History"
          name="Stroke"
          value={formData.Stroke}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Heart Disease or Heart Attack History"
          name="HeartDiseaseorAttack"
          value={formData.HeartDiseaseorAttack}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Physical Activity in Last 30 Days"
          name="PhysActivity"
          value={formData.PhysActivity}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Fruit Consumption"
          name="Fruits"
          value={formData.Fruits}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Vegetable Consumption"
          name="Veggies"
          value={formData.Veggies}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Heavy Alcohol Consumption Indicator"
          name="HvyAlcoholConsump"
          value={formData.HvyAlcoholConsump}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Healthcare Coverage"
          name="AnyHealthcare"
          value={formData.AnyHealthcare}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Could Not See Doctor Because of Cost"
          name="NoDocbcCost"
          value={formData.NoDocbcCost}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="General Health Rating"
          name="GenHlth"
          value={formData.GenHlth}
          options={generalHealthOptions}
          onChange={handleChange}
        />

        <NumberField
          label="Poor Mental Health Days in Last 30 Days"
          name="MentHlth"
          value={formData.MentHlth}
          min={0}
          max={30}
          onChange={handleChange}
        />

        <NumberField
          label="Poor Physical Health Days in Last 30 Days"
          name="PhysHlth"
          value={formData.PhysHlth}
          min={0}
          max={30}
          onChange={handleChange}
        />

        <SelectField
          label="Difficulty Walking or Climbing Stairs"
          name="DiffWalk"
          value={formData.DiffWalk}
          options={binaryOptions}
          onChange={handleChange}
        />

        <SelectField
          label="Age Group"
          name="Age"
          value={formData.Age}
          options={ageOptions}
          onChange={handleChange}
        />
      </div>

      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <button
          type="submit"
          disabled={loading}
          className="rounded-xl bg-blue-700 px-6 py-3 font-semibold text-white shadow-sm transition hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-blue-300"
        >
          {loading ? "Estimating..." : "Estimate Risk"}
        </button>

        <button
          type="button"
          onClick={handleReset}
          className="rounded-xl border border-slate-200 bg-white px-6 py-3 font-semibold text-slate-700 transition hover:bg-slate-50"
        >
          Reset Example Values
        </button>
      </div>
    </form>
  );
}