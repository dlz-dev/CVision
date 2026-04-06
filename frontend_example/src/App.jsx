import { useState, useRef } from "react"

const WEBHOOK_URL = "https://n8n.dlzteam.com/webhook/process-cv"

export default function App() {
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [filename, setFilename] = useState(null)
  const inputRef = useRef()

  async function sendFile(file) {
    if (!file || !file.name.endsWith(".txt")) {
      setError("Veuillez envoyer un fichier .txt")
      return
    }
    setFilename(file.name)
    setLoading(true)
    setResult(null)
    setError(null)
    const formData = new FormData()
    formData.append("file", file)
    try {
      const response = await fetch(WEBHOOK_URL, { method: "POST", body: formData })
      const data = await response.json()
      setResult(data)
    } catch (e) {
      setError("Erreur lors de l'envoi du fichier.")
    } finally {
      setLoading(false)
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    sendFile(e.dataTransfer.files[0])
  }

  const isInvite = result?.decision === "Inviter"

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        body {
          background: #f5f3ef;
          color: #1a1a1a;
          font-family: 'DM Sans', sans-serif;
          min-height: 100vh;
        }

        .page {
          min-height: 100vh;
          display: grid;
          grid-template-rows: auto 1fr;
        }

        .header {
          padding: 28px 56px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          border-bottom: 1px solid #e2ddd8;
          background: #faf9f7;
        }

        .logo {
          font-family: 'DM Serif Display', serif;
          font-size: 24px;
          color: #1a1a1a;
          letter-spacing: -0.5px;
        }

        .logo span { color: #8b6f47; }

        .badge {
          font-size: 11px;
          font-weight: 500;
          letter-spacing: 2px;
          text-transform: uppercase;
          color: #8b6f47;
          border: 1px solid #d4c4b0;
          padding: 6px 16px;
          border-radius: 100px;
          background: #fdf9f4;
        }

        .main {
          display: grid;
          grid-template-columns: 480px 1fr;
        }

        .left {
          padding: 72px 56px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          background: #faf9f7;
          border-right: 1px solid #e2ddd8;
        }

        .eyebrow {
          font-size: 11px;
          letter-spacing: 3px;
          text-transform: uppercase;
          color: #8b6f47;
          font-weight: 500;
          margin-bottom: 20px;
        }

        .headline {
          font-family: 'DM Serif Display', serif;
          font-size: 48px;
          line-height: 1.1;
          letter-spacing: -1px;
          color: #1a1a1a;
          margin-bottom: 20px;
        }

        .headline em {
          font-style: italic;
          color: #8b6f47;
        }

        .desc {
          font-size: 15px;
          font-weight: 300;
          line-height: 1.8;
          color: #6b6560;
          margin-bottom: 48px;
        }

        .dropzone {
          border: 1.5px dashed #d4c4b0;
          border-radius: 14px;
          padding: 44px 32px;
          text-align: center;
          cursor: pointer;
          transition: all 0.2s ease;
          background: #fff;
        }

        .dropzone:hover, .dropzone.active {
          border-color: #8b6f47;
          background: #fdf9f4;
        }

        .dropzone-icon {
          font-size: 36px;
          margin-bottom: 14px;
          display: block;
        }

        .dropzone-title {
          font-family: 'DM Serif Display', serif;
          font-size: 17px;
          color: #1a1a1a;
          margin-bottom: 6px;
        }

        .dropzone-sub {
          font-size: 13px;
          color: #a09890;
          font-weight: 300;
        }

        .spinner {
          width: 28px;
          height: 28px;
          border: 2px solid #e2ddd8;
          border-top-color: #8b6f47;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
          margin: 0 auto 14px;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        .error-msg {
          margin-top: 14px;
          padding: 12px 16px;
          background: #fdf0f0;
          border: 1px solid #f0d0d0;
          border-radius: 10px;
          font-size: 13px;
          color: #c0392b;
        }

        .right {
          padding: 72px 64px;
          overflow-y: auto;
          background: #f5f3ef;
        }

        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          gap: 14px;
          opacity: 0.3;
          text-align: center;
        }

        .empty-icon { font-size: 56px; }

        .empty-text {
          font-family: 'DM Serif Display', serif;
          font-size: 20px;
          color: #1a1a1a;
        }

        .result-filename {
          font-size: 11px;
          letter-spacing: 2px;
          text-transform: uppercase;
          color: #a09890;
          margin-bottom: 8px;
        }

        .result-name {
          font-family: 'DM Serif Display', serif;
          font-size: 36px;
          color: #1a1a1a;
          letter-spacing: -1px;
          margin-bottom: 16px;
        }

        .decision-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 9px 20px;
          border-radius: 100px;
          font-size: 13px;
          font-weight: 500;
          letter-spacing: 0.5px;
          margin-bottom: 36px;
        }

        .decision-pill.invite {
          background: #edf7f0;
          color: #2d7a4f;
          border: 1px solid #c3e6d0;
        }

        .decision-pill.reject {
          background: #fdf0f0;
          color: #c0392b;
          border: 1px solid #f0c8c8;
        }

        .dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: currentColor;
        }

        .divider {
          height: 1px;
          background: #e2ddd8;
          margin: 28px 0;
        }

        .grid-info {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
          margin-bottom: 28px;
        }

        .info-card {
          background: #fff;
          border: 1px solid #e2ddd8;
          border-radius: 12px;
          padding: 16px 20px;
        }

        .info-label {
          font-size: 10px;
          letter-spacing: 2px;
          text-transform: uppercase;
          color: #a09890;
          margin-bottom: 6px;
          font-weight: 500;
        }

        .info-value {
          font-family: 'DM Serif Display', serif;
          font-size: 20px;
          color: #1a1a1a;
        }

        .section-label {
          font-size: 10px;
          letter-spacing: 2px;
          text-transform: uppercase;
          color: #a09890;
          font-weight: 500;
          margin-bottom: 12px;
        }

        .tags {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
          margin-bottom: 28px;
        }

        .tag {
          font-size: 12px;
          padding: 5px 14px;
          border-radius: 100px;
          background: #fff;
          color: #6b6560;
          border: 1px solid #e2ddd8;
        }

        details summary {
          font-size: 11px;
          letter-spacing: 2px;
          text-transform: uppercase;
          color: #a09890;
          cursor: pointer;
          margin-bottom: 12px;
        }

        pre {
          font-size: 11px;
          color: #6b6560;
          overflow: auto;
          background: #fff;
          padding: 16px;
          border-radius: 10px;
          border: 1px solid #e2ddd8;
          max-height: 200px;
        }
      `}</style>

      <div className="page">
        <header className="header">
          <div className="logo">CV<span>ision</span></div>
          <div className="badge">Screening IA</div>
        </header>

        <div className="main">
          <div className="left">
            <p className="eyebrow">Recrutement intelligent</p>
            <h1 className="headline">
              Analysez vos CVs<br />en <em>quelques<br />secondes</em>
            </h1>
            <p className="desc">
              Déposez un CV au format .txt et obtenez instantanément une analyse structurée et une recommandation d'embauche.
            </p>

            <div
              className={`dropzone ${dragging ? "active" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".txt"
                style={{ display: "none" }}
                onChange={(e) => sendFile(e.target.files[0])}
              />
              {loading ? (
                <>
                  <div className="spinner" />
                  <div className="dropzone-title">Analyse en cours…</div>
                  <div className="dropzone-sub">{filename}</div>
                </>
              ) : (
                <>
                  <span className="dropzone-icon">📄</span>
                  <div className="dropzone-title">Déposez votre CV ici</div>
                  <div className="dropzone-sub">ou cliquez pour sélectionner un fichier .txt</div>
                </>
              )}
            </div>

            {error && <div className="error-msg">{error}</div>}
          </div>

          <div className="right">
            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">🎯</div>
                <div className="empty-text">En attente d'un CV</div>
              </div>
            ) : (
              <div>
                <div className="result-filename">{filename}</div>
                <div className="result-name">{result.name}</div>
                <div className={`decision-pill ${isInvite ? "invite" : "reject"}`}>
                  <div className="dot" />
                  {result.decision}
                </div>

                <div className="grid-info">
                  <div className="info-card">
                    <div className="info-label">Âge</div>
                    <div className="info-value">{result.age} ans</div>
                  </div>
                  <div className="info-card">
                    <div className="info-label">Expérience</div>
                    <div className="info-value">{result.total_experience_years} ans</div>
                  </div>
                  <div className="info-card">
                    <div className="info-label">Poste visé</div>
                    <div className="info-value" style={{ fontSize: "14px", paddingTop: "2px" }}>{result.target_role}</div>
                  </div>
                  <div className="info-card">
                    <div className="info-label">Diplôme</div>
                    <div className="info-value" style={{ fontSize: "14px", paddingTop: "2px" }}>{result.education?.degree}</div>
                  </div>
                </div>

                <div className="section-label">Compétences</div>
                <div className="tags">
                  {result.skills?.slice(0, 10).map((s, i) => <span key={i} className="tag">{s}</span>)}
                </div>

                <div className="section-label">Langues</div>
                <div className="tags">
                  {result.languages?.map((l, i) => <span key={i} className="tag">{l.language} — {l.level}</span>)}
                </div>

                <div className="divider" />

                <details>
                  <summary>Voir le JSON complet</summary>
                  <pre>{JSON.stringify(result, null, 2)}</pre>
                </details>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  )
}