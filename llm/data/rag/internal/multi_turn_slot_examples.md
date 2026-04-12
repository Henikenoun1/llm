# Multi Turn Slot Recovery Examples

Goal: recover intent and missing slots across multiple turns without asking many questions at once.

Case 1: order tracking
1. User: `aslema nheb suivi commande`
2. Assistant: `3atini num client loul bach nthabet el dossier.`
3. User: `3310`
4. State after turn 3:
   `active_intent=order_tracking`
   `num_client=3310`
   `missing_slots=order_id`
5. Assistant: `Mriguel. Tawa medli order_id wala num commande.`
6. User: `CMD2612345678`
7. Final extraction:
   `intent=order_tracking`
   `num_client=3310`
   `order_id=CMD2612345678`

Case 2: create order
1. User: `bonjour nheb naadi commande progressive`
2. Assistant: `Avant tout, il faut num client.`
3. User: `5100`
4. Assistant: `Mriguel. Chnia produit exactement, index, traitement, diametre?`
5. User: `progressive 1.67 crizal prevencia diametre 70`
6. Final extraction:
   `intent=create_order`
   `num_client=5100`
   `product=progressive`
   `index=1.67`
   `treatment=crizal prevencia`
   `diameter=70`

Case 3: optical values collected later
1. User: `client 5100, progressive 1.67 diametre 70`
2. Assistant: `Nethabet OD/OG qabel validation.`
3. User: `OD -1.25 +0.50 axe 90, OG -0.75 +0.25 axe 80, add +1.50`
4. Final extraction:
   `od_sphere=-1.25`
   `od_cyl=+0.50`
   `od_axis=90`
   `og_sphere=-0.75`
   `og_cyl=+0.25`
   `og_axis=80`
   `addition=+1.50`

Case 4: delivery schedule
1. User: `planning livraison agence aouina`
2. Assistant: `Chnia secteur exactement?`
3. User: `marsa lac`
4. Final extraction:
   `intent=delivery_schedule`
   `agence=Aouina`
   `secteur=marsa / lac`

Case 5: reference confirmation then availability
1. User: `confirmili 25YXSU`
2. Assistant: `Nethabet code 25YXSU men catalogue.`
3. User: `w dispo zeda?`
4. Final extraction for follow-up:
   `intent=availability_inquiry`
   `reference=25YXSU`
   `lens_code=25YXSU`

Principles:
- Keep the active business intent if the next short user reply only fills a missing slot.
- Numeric short replies can be `num_client`, `order_id`, `reference`, `quantity`, or `delivery_slot` depending on the pending state.
- Ask only for the next missing slot, not for every missing field at once.
