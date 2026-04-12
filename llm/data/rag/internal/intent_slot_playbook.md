# Intent And Slot Playbook

This document describes how Mohsen should understand user requests in SIVO call-center conversations.

Intent: `order_tracking`
- Required slots: `num_client`, `order_id`
- Optional slots: `status`, `order_type`, `customer_name`, `date`, `date_from`, `date_to`, `priority`
- Example:
  Input: `aslema, suivi commande CMD2612345678 mta3 CLI3310`
  Output intent: `order_tracking`
  Output slots: `num_client=3310`, `order_id=CMD2612345678`

Intent: `create_order`
- Required slots: `num_client`, `product`
- Important slots: `material`, `index`, `treatment`, `color`, `diameter`, `quantity`, `priority`, `reference`
- Optical slots: `od_sphere`, `od_cyl`, `od_axis`, `og_sphere`, `og_cyl`, `og_axis`, `addition`
- Example:
  Input: `bonjour, num client 5100, nheb progressive 1.67 crizal prevencia diametre 70`
  Output intent: `create_order`
  Output slots: `num_client=5100`, `product=progressive`, `index=1.67`, `treatment=crizal prevencia`, `diameter=70`

Intent: `delivery_schedule`
- Required slots: none
- Important slots: `agence`, `secteur`, `delivery_slot`
- Example:
  Input: `planning livraison agence aouina secteur marsa lac`
  Output intent: `delivery_schedule`
  Output slots: `agence=Aouina`, `secteur=marsa / lac`

Intent: `reference_confirmation`
- Required slots: `reference`
- Optional slots: `lens_code`, `product`, `material`, `diameter`
- Example:
  Input: `tnajem tconfirmili code 25YXSU`
  Output intent: `reference_confirmation`
  Output slots: `reference=25YXSU`, `lens_code=25YXSU`

Intent: `availability_inquiry`
- Required slots: none
- Useful slots: `reference`, `lens_code`, `product`, `material`, `treatment`, `diameter`, `index`, `city`
- Example:
  Input: `dispo 25YXSU orma sun diametre 65`
  Output intent: `availability_inquiry`
  Output slots: `reference=25YXSU`, `material=orma`, `diameter=65`

Intent: `get_num_client`
- Trigger when the caller starts a business request without `num_client`
- Example:
  Input: `aslema nheb suivi commande`
  Output intent: `get_num_client`
  Output slots: none

Script reminders:
- Arabic script example: `عسلامة نحب suivi commande متاع CLI3310`
- Arabizi example: `aslema nheb suivi mta3 CLI3310`
- Code-switch example: `bonjour, nheb confirmation reference 25YXSU avant creation commande`
