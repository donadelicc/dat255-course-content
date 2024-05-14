1. Deep Learning: En undergruppe av maskinlæring som involverer kunstige nevrale nettverk med flere lag, som gjør det mulig for modellen å lære intrikate mønstre og representasjoner fra data.

2. Neural Network: En beregningsmodell inspirert av strukturen og funksjonen til biologiske nevrale nettverk, sammensatt av sammenkoblede noder eller nevroner organisert i lag.

3. Activation Function: En funksjon som brukes på utgangen av hver nevron i et lag i et nevralt nettverk, som bestemmer nevronens utgang og introduserer ikke-linearitet til modellen.

4. Loss Function: En funksjon som kvantifiserer forskjellen mellom forutsagte og faktiske verdier i en maskinlæringsmodell, og tjener som et mål på modellens ytelse.

5. Gradient Descent: En optimaliseringsalgoritme brukt for å minimere loss-funksjonen ved å justere modellens parametere iterativt i retning av den bratteste nedgangen i loss-gradienten.

6. Backpropagation: En teknikk for å beregne gradientene til loss-funksjonen med hensyn til parametrene i det nevrale nettverket, som muliggjør effektiv optimalisering ved bruk av gradient descent. "Algoritmen for å bestemme hvordan et enkelt trenings eksempel ønsker å 'dytte' vektene og biasene (parametere) - ikke bare i form av om de skal opp eller ned, men i form av hvilke relative proporsjoner disse endringene forårsaker den raskeste nedgangen i kostnaden"

7. Mini-batch Gradient Descent: En variant av optimalisering hvor gradient descent-oppdateringene beregnes ved bruk av små delmengder (mini-batcher) av treningsdataene, som balanserer beregningseffektivitet og modellkonvergens.

8. Learning Rate: En hyperparameter som bestemmer steglengden på parameteroppdateringene under gradient descent, og påvirker konvergenshastigheten og stabiliteten i optimaliseringsprosessen.

9. Overfitting: Et fenomen hvor en maskinlæringsmodell presterer godt på treningsdataene, men feiler i å generalisere til ukjente data, ofte forårsaket av overdreven kompleksitet eller mangel på regularisering.

10. Regularization: Teknikker brukt for å forhindre overfitting ved å pålegge begrensninger eller straffer på modellparametrene. Hovedformålet med regularisering er å modifisere læringsalgoritmen slik at modellen generaliserer bedre. Dette oppnås ved å legge til en straff på størrelsen av koeffisientene eller modellens kompleksitet. Regularisering hjelper til med å sikre at modellen er robust og presterer godt ikke bare på treningsdataene, men også på nye, ukjente data.
    - Typer
    1. L1 (Lasso): Legger til en straff lik den absolutte verdien av størrelsen på koeffisientene. Dette kan føre til at noen koeffisienter blir nøyaktig null, som er en form for automatisk funksjonsvalg. Det er nyttig når vi tror at mange funksjoner er irrelevante eller når vi foretrekker en sparsom modell.
    2. L2 (Ridge): Legger til en straff lik kvadratet av størrelsen på koeffisientene. Dette har en tendens til å spre feilen blant alle termer og fører til mindre koeffisienter, men det eliminerer ikke nødvendigvis koeffisienter helt. Det er nyttig når vi tror at alle funksjoner har en innvirkning på utgangen.
    3. Dropout: En regulariseringsteknikk som tilfeldig setter en andel av inngangsenhetene til 0 ved hver oppdatering under trening, som hjelper til med å forhindre at nevroner ko-adapterer for mye.
    4. weight decay: En regulariseringsteknikk brukt for å forhindre overfitting i maskinlæringsmodeller ved å legge til en straffeterm til loss-funksjonen basert på størrelsen av modellens vekter. Det oppmuntrer modellen til å lære enklere og jevnere representasjoner ved å straffe store vektverdier.
    5. Early stopping: Innebærer å stoppe treningsprosessen før treningen fullføres hvis ytelsen på et valideringsdatasett begynner å forverres eller slutter å forbedres betydelig. Dette forhindrer overfitting ved å ikke la modellen trene for lenge og over-lære treningsdataene.

11. Data Augmentation: En teknikk for å øke mangfoldet i treningsdataene ved å anvende transformasjoner som rotasjon, skalering og flipping, og dermed øke modellens robusthet og generaliseringsevne.

12. Transfer Learning: En maskinlæringstilnærming hvor kunnskap oppnådd fra trening av en modell på en bestemt oppgave overføres eller tilpasses til en relatert oppgave, ofte ved å finjustere den forhåndstrente modellen.

13. Fine-tuning: Prosessen med å videre trene en forhåndstrent modell på nye data eller oppgaver, vanligvis ved å justere dens parametere mens noen av de tidligere lærte representasjonene beholdes intakte.

14. fastai: Et deep learning-bibliotek bygget på toppen av PyTorch, som gir høynivåabstraksjoner og verktøy for å forenkle utvikling og trening av nevrale nettverksmodeller.

15. DataBlock: En komponent i fastai for fleksibel og tilpassbar dataprosessering og lasting, som muliggjør sømløs integrering av ulike datakilder og formater inn i treningspipelinjen.

16. Learner: Den sentrale objektet i fastai som representerer modellen, dataene, optimaliseringsprosessen og treningsløkken, og gir et grensesnitt for trening, evaluering og inferens.

17. Image Classification: En oppgave innenfor computer vision hvor målet er å tildele en etikett eller kategori til et input-bilde basert på dets visuelle innhold, ofte utført ved bruk av deep learning-modeller.

18. Natural Language Processing (NLP): Et felt innen kunstig intelligens som fokuserer på å gjøre det mulig for datamaskiner å forstå, tolke og generere menneskelig språk, ofte ved bruk av deep learning-teknikker.

19. Transformer: En nevralt nettverksarkitektur introdusert i "Attention is All You Need" artikkelen av Google, mye brukt i NLP-oppgaver for sine paralleliseringsevner og effektivitet i å fange langdistanseavhengigheter.

20. GPT (Generative Pre-trained Transformer): En serie av transformer-baserte modeller utviklet av OpenAI for oppgaver innen naturlig språk generering, som utnytter storskala usupervisert forhåndstrening etterfulgt av finjustering på spesifikke oppgaver.

21. Computer Vision: Et felt innen kunstig intelligens som fokuserer på å gjøre det mulig for datamaskiner å tolke og analysere visuell informasjon fra den virkelige verden, ofte ved bruk av deep learning-teknikker for oppgaver som objektdeteksjon og bildesegmentering.

22. Object Detection: En computer vision-oppgave som involverer identifikasjon og lokalisering av objekter i et bilde, ofte oppnådd ved å forutsi rammebokser og klasseetiketter for hvert objekt.

23. Segmentation: En computer vision-oppgave hvor målet er å dele et bilde inn i flere segmenter eller regioner, vanligvis representerer forskjellige objekter eller interesseområder, ofte brukt i medisinsk bildebehandling og scene forståelse.

24. Reinforcement Learning: Et maskinlæringsparadigme hvor en agent lærer å ta beslutninger ved å samhandle med et miljø, motta belønninger eller straffer basert på sine handlinger, ofte brukt i spill og robotikk.

25. fine_tune() method in fastai: En metode eller funksjon som ofte finnes i deep learning-biblioteker som fastai, brukt for videre trening av en forhåndstrent nevralt nettverksmodell på et nytt datasett eller oppgave. Finjustering innebærer typisk å justere parametrene til den forhåndstrente modellen mens noen av de tidligere lærte representasjonene beholdes intakte, og dermed utnytte kunnskapen oppnådd fra forhåndstreningen for å forbedre ytelsen på den nye oppgaven.

    - Hva den gjør
    1. Pretrained Model Usage: Starter med en modell som allerede er trent på et annet datasett (vanligvis et stort og generisk datasett som ImageNet).
    2. Freeze Training: Opprinnelig er alle lagene i modellen bortsett fra de siste få frosset. Dette betyr at vektene deres ikke oppdateres under den første fasen av treningen.
    3. Training the New Layers: De ufryste lagene trenes deretter på det nye datasettet for et spesifisert antall epoker.
    4. Unfreezing and Fine-Tuning: Valgfritt, etter den innledende treningen av de nye lagene, kan hele modellen bli ufryst og alle lagene kan bli finjustert sammen for ytterligere epoker.
    - Viktigste parametere
    1. epochs: Totalt antall epoker for å trene modellen. Dette deles inn i to faser - innledende trening av de nye lagene og full modell finjustering.
    2. base_lr: Læringsraten som skal anvendes. FastAI kan anvende denne læringsraten i henhold til sin diskriminerende finjusteringsfilosofi, hvor forskjellige lag kan ha forskjellige læringsrater.
    3. freeze_epochs: Antall epoker hvor de forhåndstrente lagene er frosset og kun de siste få lagene trenes.

26. DataBlock: I konteksten av fastai er DataBlock en høynivåabstraksjon for å definere dataprosesseringspipelinjen i maskinlæringsoppgaver. Det lar brukere spesifisere hvordan man skal transformere rådata til et egnet format, f.eks. bygge et datasett. DataBlock gir fleksibilitet og tilpasningsmuligheter for å forbehandle data og opprette DataLoaders.

27. DataLoaders: DataLoaders er objekter som brukes til effektiv lasting og iterering over batcher av data under trenings-, validerings- og testfaser av en maskinlæringsoppgave. I fastai opprettes DataLoaders vanligvis ved bruk av DataBlock API, og kapsler inn trenings- og valideringsdatasett sammen med dataforsterkning, batching, shuffling og andre dataprosesseringsoperasjoner. De gir et praktisk grensesnitt for å mate data til treningsløkken.

28. Embedding: I konteksten av deep learning refererer en embedding til en kartlegging fra diskrete kategoriske variabler (som ord, tokens eller kategorier) til kontinuerlige vektorrepresentasjoner i et lavere dimensjonsrom. Embeddings læres under treningsprosessen og fanger semantiske relasjoner mellom kategoriske verdier, som gjør det mulig for nevrale nettverk å effektivt behandle og generalisere fra kategoriske inndata.

29. Learning Rate: Læringsraten er en hyperparameter som styrer steglengden på parameteroppdateringer under treningsprosessen av et nevralt nettverk. Det bestemmer hvor mye modellens parametere justeres i retning av gradienten under optimalisering. En passende læringsrate er avgjørende for å oppnå optimal konvergens og ytelse i trening av nevrale nettverk, med verdier vanligvis valgt gjennom eksperimentering og validering.

30. Learning Rate Finder: En teknikk brukt for å bestemme en passende læringsrate for trening av nevrale nettverk, spesielt i konteksten av stochastic gradient descent-optimaliseringsalgoritmer. Learning rate finner involverer gradvis økning av læringsraten i de tidlige stadiene av treningen mens man overvåker oppførselen til loss-funksjonen. Den optimale læringsraten identifiseres vanligvis som punktet hvor loss begynner å avta mest raskt eller flater ut.

    - Hvordan det fungerer
    1. Initial Setup: Learning rate finner starter med en veldig liten læringsrate og øker den gradvis over en serie iterasjoner.
    2. Training Mini-batches: For hver mini-batch trenes modellen med den nåværende læringsraten, og læringsraten økes deretter eksponentielt.
    3. Tracking Loss: Den registrerer tapet ved hvert trinn. Ideen er å se hvordan tapet endrer seg etter hvert som læringsraten øker.
    4. Plotting Loss: Etter å ha kjørt gjennom mini-batchene, plottes tapet mot læringsratene.

31. Metric: En metrikk er et mål brukt for å evaluere ytelsen til en maskinlæringsmodell på en spesifikk oppgave eller datasett. Metrikker kvantifiserer ulike aspekter av modellens ytelse, som nøyaktighet, presisjon, recall, F1-score, gjennomsnittlig kvadratisk feil eller areal under mottakeroperasjonskurven (ROC-AUC). Metrikker er essensielle for å vurdere effektiviteten og kvaliteten til trente modeller og veilede modellvalg og optimaliseringsprosesser.

    - Oppgaver og deres mest brukte metrikk
    1. Klassifisering
        Nøyaktighet: Andelen riktige forutsigelser (både sanne positive og sanne negative) blant det totale antallet undersøkte tilfeller.
        Cross Entropy Loss (Log Loss): En ytelsesmål for å evaluere sannsynlighetene som en klassifiserer gir i motsetning til dens diskrete forutsigelser. Lavere log loss-verdier indikerer bedre ytelse, med perfekte modeller som har en log loss på null.
    2. Klassifisering, spesielt med ubalanserte datasett
        Presisjon og Recall
            Presisjon: Forholdet mellom sanne positive forutsigelser og det totale antallet forutsagte positive - nøyaktigheten av positive forutsigelser.
            Recall: Forholdet mellom sanne positive forutsigelser og de faktiske positive tilfellene - evnen til å oppdage positive tilfeller.
    3. Klassifiseringsoppgaver hvor balansering av presisjon og recall er viktig.
        F1 Score: Den harmoniske gjennomsnittsverdien av presisjon og recall. Det er spesielt nyttig når du trenger en balanse mellom presisjon og recall, og det er en ujevn klassedistribusjon.
    4. Regressjon (forutsi et resultat):
        MAE: Gjennomsnittet av de absolutte forskjellene mellom de forutsagte verdiene og de faktiske verdiene. Det gir en idé om hvor feil forutsigelsene var; jo lavere MAE, desto bedre.
        MSE: Gjennomsnittet av kvadratene av forskjellene mellom de forutsagte verdiene og de faktiske verdiene. Det straffer større feil mer enn MAE.

32. Generative AI: Generative AI refererer til kunstige intelligenssystemer og algoritmer som er i stand til å generere nytt innhold, som bilder, tekst, lyd eller video, som etterligner eller ligner menneskeskapt data. Generative AI-modeller lærer ofte å fange og replikere den underliggende fordelingen av treningsdata, som gjør det mulig for dem å produsere nye og realistiske utganger.

32. Collaborative Filtering: Collaborative filtering er en teknikk brukt i anbefalingssystemer for å forutsi en brukers preferanser eller interesser basert på preferansene til lignende brukere eller gjenstander. Den utnytter den kollektive visdommen til en gruppe brukere for å gi personlige anbefalinger, vanligvis ved å analysere bruker-gjenstands interaksjonsdata som rangeringer, anmeldelser eller kjøpshistorikk. Collaborative filtering kan implementeres ved hjelp av forskjellige tilnærminger, inkludert brukerbasert filtrering, gjenstandsbasert filtrering og matrix-faktorisering metoder.

33. ResNet: ResNet, kort for Residual Network, er en dyp nevralt nettverksarkitektur foreslått av Kaiming He et al. i deres 2015 artikkel "Deep Residual Learning for Image Recognition." ResNet introduserte konseptet residual blocks, hvor inngangen til en blokk legges til utgangen, som gjør det lettere å trene svært dype nevrale nettverk. ResNet-arkitekturer har oppnådd toppmoderne ytelse i ulike computer vision-oppgaver, inkludert bildeklassifisering, objektdeteksjon og semantisk segmentering.


33. Hyperparameters: Hyperparametere er parametere hvis verdier settes før treningsprosessen begynner og forblir konstante under trening. Eksempler inkluderer læringsrate, batch-størrelse og antall lag i et nevralt nettverk. Justering av hyperparametere er en avgjørende del av optimalisering av ytelsen til maskinlæringsmodeller.

34. Resize(): En metode som ofte brukes i bildebehandling og computer vision-oppgaver for å endre størrelsen på bilder til en spesifisert størrelse. Det brukes til å standardisere dimensjonene til bilder i et datasett, og sikrer at alle bilder har samme bredde og høyde, noe som ofte er nødvendig for å mate dem inn i nevrale nettverk for trening eller inferens.

35. item_tfms(): I biblioteker som fastai, refererer "item_tfms()" til transformasjoner som anvendes på individuelle elementer eller prøver i et datasett. Disse transformasjonene inkluderer vanligvis resizing, cropping, normalisering og andre forbehandlingssteg utført på hvert element før de brukes til trening eller inferens.

36. batch_tfms(): I likhet med "item_tfms()", refererer "batch_tfms()" i fastai til transformasjoner som anvendes på batcher av data under treningsprosessen. Disse transformasjonene brukes vanligvis etter dataforsterkning og er nyttige for oppgaver som datanormalisering, dataforsterkning og regularisering.

37. batch: En batch er en delmengde av datasamples fra et datasett som behandles sammen under trening eller inferens. Batching tillater mer effektiv beregning, spesielt på maskinvareakseleratorer som GPUer, ved å parallellisere operasjoner på tvers av flere datasamples.

38. Confusion matrix: En forvirringsmatrise er en tabell brukt til å evaluere ytelsen til en klassifikasjonsmodell. Den presenterer et sammendrag av de forutsagte og faktiske klasselabelene for et klassifikasjonsproblem, og viser antall sanne positive, sanne negative, falske positive og falske negative.

39. Gradient: I konteksten av optimaliseringsalgoritmer representerer gradienten retningen og størrelsen av den bratteste økningen av en funksjon. Den peker i retningen av den største økningen i funksjonens verdi, og dens størrelse indikerer endringsraten.

40. SGD: SGD står for Stochastic Gradient Descent, en optimaliseringsalgoritme som ofte brukes i trening av maskinlæringsmodeller. Den oppdaterer modellparametrene iterativt ved å beregne gradienter på små tilfeldige delmengder av treningsdataene, noe som gjør den beregningseffektiv for store datasett.

41. Tokenization: Tokenization er prosessen med å dele tekstdata i mindre enheter kalt tokens. Tokens kan være ord, delord, tegn eller andre meningsfulle enheter av tekst, avhengig av oppgaven og tokenizeren som brukes.

42. gradual unfreezing: Gradvis opplåsing er en teknikk brukt i transfer learning, hvor lagene i en forhåndstrent modell låses opp gradvis under trening. Dette gjør det mulig for modellen å tilpasse seg den nye oppgaven mens de lærte representasjonene i de tidlige lagene bevares.

43. numericalization: Numerisering er prosessen med å konvertere kategoriske data til numerisk form. Dette er avgjørende for å stabilisere og akselerere treningen av nevrale nettverk.

    - Typer
    1. One-hot Encoding: En forbehandlingssteg som innebærer å konvertere hver kategoriske variabel til en ny binær variabel for hver kategori.
    2. Batch Normalization: Normaliserer aktiveringene av et tidligere lag ved å trekke fra batch-gjennomsnittet og dele på 3. batch-standardavviket.

44. Ensambling: Ensemblemetoder er teknikker i maskinlæring som kombinerer flere modeller for å produsere en bedre ytende modell enn de individuelle modellene som utgjør ensemblet.
    - Fordeler
    1. Redusere overfitting: Ulike modeller vil sannsynligvis overfitte forskjellige aspekter av dataene, og aggregering kan bidra til å gjennomsnittliggjøre disse effektene.
    3. Forbedre generalisering: Ensemblemodeller presterer ofte bedre på ukjente data sammenlignet med enkeltmodeller.
    - Typer
    1. Bagging: Innebærer å trene flere modeller uavhengig på forskjellige delmengder av dataene. Hver modell lærer fra en litt annen versjon av treningsdataene. Etter trening kan prediksjonene fra hver nettverk gjennomsnittliggjøres (for regresjon) eller stemmes på (for klassifisering) for å produsere endelig utgang.
    2. Boosting: Her kombineres flere svake lærere sekvensielt for å skape en sterk lærer. Hver ny lærer fokuserer på feilene gjort av de forrige, og forbedrer gradvis den samlede ytelsen til modellen.
    3. Stacking: Ulike modeller trenes på samme datasett, og en ny modell, kalt en meta-modell eller blender, trenes deretter for å gjøre en endelig prediksjon basert på prediksjonene til de forrige modellene. Basemodellens prediksjoner brukes som inndatafunksjoner for meta-modellen, som deretter lærer hvordan man best kan kombinere disse prediksjonene for å gjøre den endelige prediksjonen.

45. What is a latent factor? Why is it "latent"?: En latent faktor refererer til en variabel som ikke er direkte observert, men er inferert eller estimert fra andre observerte variabler innenfor en modell. Disse faktorene brukes vanligvis til å forklare mønstre i dataene som ikke er umiddelbart synlige, og de spiller en avgjørende rolle i ulike typer analyse, inkludert faktoranalyse, principal component analysis (PCA), og anbefalingssystemer.

44. embedding matrix (embeddings): Embedding-matrisen er essensielt en oppslagstabell, vanligvis brukt for å transformere høy-dimensjonal kategorisk data til et lavere dimensjonsrom. Hver rad i matrisen tilsvarer en spesifikk kategori (som et ord i NLP). Når en kategorisk enhet må konverteres til en embedding, innebærer prosessen å hente en rad fra matrisen. Denne raden er den innebygde representasjonen av den kategoriske enheten og læres fra data under treningsprosessen av en maskinlæringsmodell.

45. Embeddings vs. One-hot Encoding
    - Effektivitet: Embeddings reduserer dimensjonaliteten betydelig, noe som reduserer modellens kompleksitet og forbedrer beregningseffektiviteten.
    - Semantisk Representasjon: Embeddings fanger mer informasjon og relasjoner mellom kategorier (som semantisk likhet mellom ord) enn one-hot vektorer.
    - Trenbarhet: I motsetning til statiske one-hot vektorer, er embeddings lært og optimalisert under trening, slik at de kan tilpasse seg den spesifikke oppgaven ved å fange relevante mønstre i dataene.

46. bootstrapping problem: Bootstrapping-problemet refererer til utfordringen med å estimere påliteligheten av forutsigelser gjort av en modell når modellens treningsdata er begrenset eller støyende. I anbefalingssystemer refererer det til den innledende utfordringen med å generere nyttige anbefalinger når det er utilstrekkelig data tilgjengelig.

    - Problemer
    1. Cold Start Problem: Bootstrapping-problemet er ofte knyttet til cold start-problemet, hvor systemet har lite eller ingen informasjon om nye brukere eller nye gjenstander. Denne mangelen på data gjør det vanskelig å anbefale elementer som brukeren kanskje liker.
    2. Data Scarcity: Tidlig i implementeringen av et anbefalingssystem er det vanligvis begrenset interaksjonsdata (som rangeringer, klikk, visninger, kjøp). Denne mangelen på data hemmer systemets evne til å lære og generalisere brukerpreferanser effektivt.
    - Løsninger
    1. Content-Based Filtering: Bruker informasjon om elementene (f.eks. metadata som sjanger, beskrivelse, etc.) for å lage anbefalinger, noe som hjelper til med å lindre item cold start-problemet.
    2. Explicit Feedback: Be om at nye brukere gir innledende preferanser eller vurderer noen få elementer for raskt å samle data om deres liker og misliker.
    3. Implicit Feedback: Bruker indirekte signaler som surfehistorikk, klikkmønstre eller tid brukt på forskjellige elementer for å utlede brukerpreferanser.
    4. Demographic-Based Recommendations: Bruker demografisk informasjon (alder, kjønn, plassering) for å anbefale elementer som er populære blant lignende demografiske grupper.

47. feature: I maskinlæring refererer en funksjon til en individuell målbar egenskap eller karakteristikk av en datasample som brukes som input til en modell for å lage forutsigelser eller klassifikasjoner. Funksjoner kan være numeriske, kategoriske eller ordinale, og de fanger relevant informasjon om datasample.
